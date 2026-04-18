"""
FastAPI server — LiDAR + Camera Fusion (v2).
Exposes /infer, /infer-bulk, /rag-query, /scenes, /query, /chat.
"""
import io
import os
import time
import uuid
import zipfile

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat_router import router as chat_router
from modules.loader import load_scene, normalize_calib_dict
from modules.calibration import parse_calib
from modules.fusion_pp import run_fused_pipeline
from modules.visualizer import render_lidar_bev_white, cv2_to_base64, generate_bev
from modules.chroma_store import store_scene, query_scenes
from modules.synthetic import generate_synthetic_scene, get_synthetic_detections
from modules.label_parser import parse_label_file
from modules.metrics import match_and_evaluate
from modules.bulk import process_zip

app = FastAPI(title="LiDAR Fusion API v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", os.environ.get("FRONTEND_URL", "")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


# ── Calib helper ───────────────────────────────────────────────────────────────

def _build_calib_with_proj(calib_raw: dict) -> dict:
    """
    Parse calib and add T_velo_to_img = P2 @ R0_rect @ Tr_velo_to_cam.
    Required by fusion_pp (notebook pipeline).
    """
    calib_raw  = normalize_calib_dict(calib_raw)
    calib      = parse_calib(calib_raw)          # {P2, R0_rect, Tr_velo_to_cam}
    T_velo_to_img = calib["P2"] @ calib["R0_rect"] @ calib["Tr_velo_to_cam"]
    calib["T_velo_to_img"] = T_velo_to_img
    return calib


# ── Main inference ─────────────────────────────────────────────────────────────

@app.post("/infer")
async def infer(
    bin_file:   UploadFile = File(None),
    image_file: UploadFile = File(None),
    calib_file: UploadFile = File(None),
    label_file: UploadFile = File(None),
):
    """
    Full LiDAR + camera fusion (PP + YOLO + DBSCAN/OBB).
    Falls back to synthetic scene if no files uploaded.
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
    calib = _build_calib_with_proj(scene["calib"])

    # Convert RGB image from loader to BGR for cv2 in fusion_pp
    import cv2
    image_bgr = cv2.cvtColor(scene["image"], cv2.COLOR_RGB2BGR)

    img_lidar, img_boxes, serial_dets, stats = run_fused_pipeline(
        scene["points"], image_bgr, calib
    )

    # White-background BEV (matches image.png style)
    lidar_bev = render_lidar_bev_white(scene["points"][:, :3], serial_dets)

    # Unique frame ID for ChromaDB
    frame_id = f"frame_{uuid.uuid4().hex[:8]}"

    # Store in ChromaDB for RAG
    store_scene(frame_id, serial_dets, int(len(scene["points"])))

    # Optional GT evaluation
    ground_truth = None
    eval_metrics  = None
    if label_file:
        label_bytes  = await label_file.read()
        # Convert serial_dets to format expected by metrics (use class + bbox_2d)
        pred_for_metrics = [
            {
                "class":      d["label"],
                "confidence": d["score"],
                "bbox_2d":    d["bbox_2d"],
                "distance_m": d["distance_m"],
            }
            for d in serial_dets
        ]
        ground_truth = parse_label_file(label_bytes)
        eval_metrics = match_and_evaluate(pred_for_metrics, ground_truth)

    return {
        "camera_image":    cv2_to_base64(img_boxes),
        "lidar_image":     cv2_to_base64(img_lidar),
        "lidar_bev":       lidar_bev,
        "detections":      serial_dets,
        "pipeline_stats":  stats,
        "ground_truth":    ground_truth,
        "metrics":         eval_metrics,
        "inference_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "num_points":      int(len(scene["points"])),
        "frame_id":        frame_id,
    }


# ── Scenes ─────────────────────────────────────────────────────────────────────

KITTI_DATA_DIR = os.environ.get("KITTI_DATA_DIR", "data/kitti")
DUMMY_SCENES   = ["scene_0001", "scene_0002", "scene_0003", "scene_0007", "scene_0015"]


@app.get("/scenes")
async def list_scenes():
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
    t0   = time.perf_counter()
    seed = hash(scene_id) % (2 ** 31)

    scene_path = os.path.join(KITTI_DATA_DIR, scene_id)
    if os.path.isdir(scene_path):
        import glob as _glob
        bin_files   = _glob.glob(os.path.join(scene_path, "velodyne", "*.bin"))
        img_files   = _glob.glob(os.path.join(scene_path, "image_02", "*.png"))
        calib_files = _glob.glob(os.path.join(scene_path, "*.txt"))
        if bin_files and img_files and calib_files:
            bin_bytes   = open(bin_files[0],   "rb").read()
            img_bytes   = open(img_files[0],   "rb").read()
            calib_bytes = open(calib_files[0], "rb").read()
        else:
            bin_bytes, img_bytes, calib_text = generate_synthetic_scene(seed=seed)
            calib_bytes = calib_text.encode("utf-8")
    else:
        bin_bytes, img_bytes, calib_text = generate_synthetic_scene(seed=seed)
        calib_bytes = calib_text.encode("utf-8")

    scene = load_scene(bin_bytes, img_bytes, calib_bytes)
    calib = _build_calib_with_proj(scene["calib"])

    import cv2
    image_bgr = cv2.cvtColor(scene["image"], cv2.COLOR_RGB2BGR)
    img_lidar, img_boxes, serial_dets, stats = run_fused_pipeline(
        scene["points"], image_bgr, calib
    )
    lidar_bev = render_lidar_bev_white(scene["points"][:, :3], serial_dets)
    store_scene(scene_id, serial_dets, int(len(scene["points"])))

    return {
        "camera_image":      cv2_to_base64(img_boxes),
        "lidar_image":       cv2_to_base64(img_lidar),
        "lidar_bev":         lidar_bev,
        "detections":        serial_dets,
        "pipeline_stats":    stats,
        "inference_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "num_points":        int(len(scene["points"])),
        "frame_id":          scene_id,
    }


# ── Bulk ZIP inference ─────────────────────────────────────────────────────────

@app.post("/infer-bulk")
async def infer_bulk(
    zip_file:   UploadFile = File(...),
    max_frames: int = 20,
):
    zip_bytes = await zip_file.read()
    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        raise HTTPException(status_code=400, detail="Not a valid ZIP archive.")

    result = process_zip(zip_bytes, max_frames=max_frames)

    # Store each bulk frame in ChromaDB
    for frame in result.get("frames", []):
        fid  = frame.get("frame_id", f"bulk_{uuid.uuid4().hex[:6]}")
        dets = frame.get("detections", [])
        npts = frame.get("num_points", 0)
        store_scene(fid, dets, npts)

    return result


# ── RAG query ─────────────────────────────────────────────────────────────────

class RagRequest(BaseModel):
    text: str
    n_results: int = 5


@app.post("/rag-query")
async def rag_query(req: RagRequest):
    """
    Semantic search over stored scenes, then generate answer via OpenRouter.
    """
    matches = query_scenes(req.text, n_results=req.n_results)
    if not matches:
        return {"answer": "No scenes stored yet. Run inference on some frames first.", "matches": []}

    context = "\n".join(m["text"] for m in matches)

    # Build answer via OpenRouter
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    answer  = ""
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            resp = client.chat.completions.create(
                model="anthropic/claude-sonnet-4-6",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an autonomous vehicle perception analyst. "
                            "Answer the user's question about KITTI LiDAR scenes "
                            "using only the provided context.\n\n"
                            f"Context (top {len(matches)} matching scenes):\n{context}"
                        ),
                    },
                    {"role": "user", "content": req.text},
                ],
                max_tokens=512,
            )
            answer = resp.choices[0].message.content
        except Exception as exc:
            answer = f"[RAG LLM error: {exc}]\n\nRelevant scenes:\n{context}"
    else:
        answer = f"(No OPENROUTER_API_KEY set — raw matches below)\n\n{context}"

    return {"answer": answer, "matches": matches}


# ── Semantic query stub ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str
    max_distance_m: float | None = None


@app.post("/query")
async def query_scene(req: QueryRequest):
    objects  = get_synthetic_detections(seed=42, n=6)
    keywords = req.text.lower().split()
    results  = [
        d for d in objects
        if any(k in d["class"].lower() for k in keywords)
        or any(k in ("car", "vehicle", "object") for k in keywords)
    ]
    if req.max_distance_m is not None:
        results = [d for d in results if d.get("distance_m", 999) <= req.max_distance_m]
    return {"results": results, "query": req.text}
