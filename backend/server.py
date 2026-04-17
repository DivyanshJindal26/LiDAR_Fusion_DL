"""
FastAPI server for LiDAR + Camera Fusion demo.
Exposes /infer, /scenes, /query, and /chat endpoints.
"""
import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat_router import router as chat_router
from modules.loader import load_scene
from modules.calibration import parse_calib
from modules.detector import detect
from modules.fusion import fuse
from modules.fusion_b import fuse_b
from modules.visualizer import annotate_image, generate_bev, bbox_to_frustum_corners
from modules.synthetic import generate_synthetic_scene, get_synthetic_detections
from modules.label_parser import parse_label_file
from modules.metrics import match_and_evaluate

app = FastAPI(title="LiDAR Fusion API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", os.environ.get("FRONTEND_URL", "")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


# ── Inference ──────────────────────────────────────────────────────────────────

@app.post("/infer")
async def infer(
    bin_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
    calib_file: UploadFile = File(None),
    label_file: UploadFile = File(None),
):
    """
    Full LiDAR + camera fusion pipeline.
    - If label_file (.txt) is provided: uses ground-truth KITTI labels, skips YOLOv8.
    - If only bin+image+calib: runs YOLOv8 + LiDAR fusion.
    - If nothing uploaded: generates a synthetic scene.
    """
    t0 = time.perf_counter()

    if bin_file and image_file and calib_file:
        bin_bytes   = await bin_file.read()
        img_bytes   = await image_file.read()
        calib_bytes = await calib_file.read()
    else:
        bin_bytes, img_bytes, calib_text = generate_synthetic_scene(seed=42)
        calib_bytes = calib_text.encode("utf-8")

    scene = load_scene(bin_bytes, img_bytes, calib_bytes)
    calib_parsed = parse_calib(scene["calib"])

    # Always run the model — use Approach B (RANSAC + DBSCAN + PCA)
    detections_2d = detect(scene["image"])
    detections    = fuse_b(detections_2d, scene["points"], calib_parsed, scene["image"].shape[:2])

    # Annotate each detection with projected 2D corners for frontend 3D rendering
    P2 = calib_parsed.get("P2")
    for det in detections:
        if P2 is not None and det.get("box_3d") and det.get("bbox_2d"):
            length = float(det["box_3d"][5]) if det["box_3d"][5] > 0.1 else 4.0
            det["corners_2d"] = bbox_to_frustum_corners(
                det["bbox_2d"], det["distance_m"], length, P2)
        else:
            det["corners_2d"] = None

    # If GT labels provided, compute metrics against predictions
    ground_truth = None
    eval_metrics = None
    if label_file:
        label_bytes  = await label_file.read()
        ground_truth = parse_label_file(label_bytes)
        eval_metrics = match_and_evaluate(detections, ground_truth)

    annotated = annotate_image(scene["image"], detections, calib_parsed, ground_truth)
    bev = generate_bev(scene["points"], detections)

    return {
        "annotated_image":   annotated,
        "bev_image":         bev,
        "detections":        detections,
        "ground_truth":      ground_truth,
        "metrics":           eval_metrics,
        "inference_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "num_points":        len(scene["points"]),
    }


# ── Scenes ─────────────────────────────────────────────────────────────────────

KITTI_DATA_DIR = os.environ.get("KITTI_DATA_DIR", "data/kitti")
DUMMY_SCENES = ["scene_0001", "scene_0002", "scene_0003", "scene_0007", "scene_0015"]


@app.get("/scenes")
async def list_scenes():
    """Return list of scene IDs (real KITTI dirs if available, else synthetic stubs)."""
    try:
        real = sorted(
            d for d in os.listdir(KITTI_DATA_DIR)
            if os.path.isdir(os.path.join(KITTI_DATA_DIR, d))
        )
        return real if real else DUMMY_SCENES
    except FileNotFoundError:
        return DUMMY_SCENES


@app.post("/infer-scene/{scene_id}")
async def infer_scene(scene_id: str):
    """Run inference on a scene by ID (uses synthetic data seeded by scene name)."""
    t0 = time.perf_counter()
    seed = hash(scene_id) % (2 ** 31)

    scene_path = os.path.join(KITTI_DATA_DIR, scene_id)
    if os.path.isdir(scene_path):
        # Load real KITTI files if available
        import glob as _glob
        bin_files = _glob.glob(os.path.join(scene_path, "velodyne", "*.bin"))
        img_files = _glob.glob(os.path.join(scene_path, "image_02", "*.png"))
        calib_files = _glob.glob(os.path.join(scene_path, "*.txt"))
        if bin_files and img_files and calib_files:
            bin_bytes = open(bin_files[0], "rb").read()
            img_bytes = open(img_files[0], "rb").read()
            calib_bytes = open(calib_files[0], "rb").read()
        else:
            bin_bytes, img_bytes, calib_text = generate_synthetic_scene(seed=seed)
            calib_bytes = calib_text.encode("utf-8")
    else:
        bin_bytes, img_bytes, calib_text = generate_synthetic_scene(seed=seed)
        calib_bytes = calib_text.encode("utf-8")

    scene = load_scene(bin_bytes, img_bytes, calib_bytes)
    calib_parsed = parse_calib(scene["calib"])
    detections_2d = detect(scene["image"])
    detections = fuse_b(detections_2d, scene["points"], calib_parsed, scene["image"].shape[:2])
    P2 = calib_parsed.get("P2")
    for det in detections:
        if P2 is not None and det.get("box_3d") and det.get("bbox_2d"):
            length = float(det["box_3d"][5]) if det["box_3d"][5] > 0.1 else 4.0
            det["corners_2d"] = bbox_to_frustum_corners(
                det["bbox_2d"], det["distance_m"], length, P2)
        else:
            det["corners_2d"] = None
    annotated = annotate_image(scene["image"], detections, calib_parsed)
    bev = generate_bev(scene["points"], detections)

    return {
        "annotated_image":  annotated,
        "bev_image":        bev,
        "detections":       detections,
        "inference_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "num_points":        len(scene["points"]),
    }


# ── Vector store query ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str
    max_distance_m: float | None = None


@app.post("/query")
async def query_scene(req: QueryRequest):
    """Semantic search stub — returns synthetic detections filtered by keyword."""
    objects = get_synthetic_detections(seed=42, n=6)
    keywords = req.text.lower().split()
    results = [
        d for d in objects
        if any(k in d["class"].lower() for k in keywords)
        or any(k in ("car", "vehicle", "object") for k in keywords)
    ]
    if req.max_distance_m is not None:
        results = [d for d in results if d.get("distance_m", 999) <= req.max_distance_m]
    return {"results": results, "query": req.text}
