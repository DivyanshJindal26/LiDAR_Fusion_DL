"""
Bulk KITTI dataset processor.

Accepts a ZIP file, auto-detects KITTI format (object-detection per-frame calib
or raw date-level calib), extracts all valid frames, runs the full pipeline on
each one, and returns a list of per-frame results.

Supported ZIP structures
------------------------
Object detection (per-frame calib):
    [root/]velodyne/000000.bin
    [root/]image_2/000000.png
    [root/]calib/000000.txt

KITTI raw (date-level calib):
    [date/drive_sync/]velodyne_points/data/0000000000.bin
    [date/drive_sync/]image_02/data/0000000000.png
    [date/]calib_cam_to_cam.txt
    [date/]calib_velo_to_cam.txt
"""
import io
import zipfile
from pathlib import PurePosixPath

import numpy as np

from modules.loader import parse_calib_text, normalize_calib_dict, load_scene
from modules.calibration import parse_calib
from modules.detector import detect
from modules.fusion_b import fuse_b
from modules.visualizer import annotate_image, generate_bev, project_box3d, bbox_to_frustum_corners


# ── ZIP content discovery ──────────────────────────────────────────────────────

def _categorise(names: list[str]) -> dict:
    """Split ZIP member names into categorised dicts by type."""
    cats: dict = {
        "bin": {},   # stem → zip path
        "png": {},   # stem → zip path   (prefers image_02 / image_2)
        "calib_frame": {},  # stem → zip path   (per-frame calib.txt)
        "calib_cam": None,  # zip path of calib_cam_to_cam.txt
        "calib_velo": None, # zip path of calib_velo_to_cam.txt
    }
    for name in names:
        p = PurePosixPath(name)
        stem = p.stem
        low = name.lower()

        if p.suffix == ".bin":
            cats["bin"][stem] = name

        elif p.suffix == ".png":
            # Prefer the left-colour camera directory
            if "image_02" in low or "image_2" in low:
                cats["png"][stem] = name
            elif stem not in cats["png"]:
                cats["png"][stem] = name

        elif p.suffix == ".txt":
            if "calib_cam_to_cam" in low:
                cats["calib_cam"] = name
            elif "calib_velo_to_cam" in low:
                cats["calib_velo"] = name
            elif "calib" in low or (
                # object-detection format: calib/000000.txt
                len(p.parts) >= 2 and "calib" in p.parts[-2].lower()
            ):
                cats["calib_frame"][stem] = name

    return cats


def _build_calib_dict(zf: zipfile.ZipFile, cats: dict, stem: str) -> dict | None:
    """
    Return a normalised calib dict for a frame, or None if calib is missing.
    Tries per-frame calib first, then date-level cam_to_cam + velo_to_cam.
    """
    # Per-frame calib (object-detection format)
    if stem in cats["calib_frame"]:
        raw = parse_calib_text(zf.read(cats["calib_frame"][stem]).decode("utf-8", errors="replace"))
        return normalize_calib_dict(raw)

    # Date-level calib (raw format) — merge both files
    if cats["calib_cam"] and cats["calib_velo"]:
        raw = {}
        raw.update(parse_calib_text(zf.read(cats["calib_cam"]).decode("utf-8", errors="replace")))
        raw.update(parse_calib_text(zf.read(cats["calib_velo"]).decode("utf-8", errors="replace")))
        return normalize_calib_dict(raw)

    # Single date-level cam_to_cam only (no Tr_velo_to_cam → pipeline will fail)
    if cats["calib_cam"]:
        raw = parse_calib_text(zf.read(cats["calib_cam"]).decode("utf-8", errors="replace"))
        return normalize_calib_dict(raw)

    return None


# ── Per-frame pipeline ─────────────────────────────────────────────────────────

def _process_frame(bin_bytes: bytes, img_bytes: bytes, calib_dict: dict) -> dict:
    import time
    t0 = time.perf_counter()

    scene = load_scene(bin_bytes, img_bytes, b"")  # we pass calib separately
    scene["calib"] = calib_dict                    # overwrite with normalised dict

    calib_parsed = parse_calib(scene["calib"])
    detections_2d = detect(scene["image"])
    detections = fuse_b(detections_2d, scene["points"], calib_parsed, scene["image"].shape[:2])

    P2 = calib_parsed.get("P2")
    for det in detections:
        corners_2d = None
        if P2 is not None and det.get("box_3d"):
            pts = project_box3d(det["box_3d"], P2)
            if pts is not None:
                near = pts[4:8].tolist()
                far  = pts[0:4].tolist()
                corners_2d = near + far
        if corners_2d is None and P2 is not None and det.get("box_3d") and det.get("bbox_2d"):
            length = float(det["box_3d"][5]) if det["box_3d"][5] > 0.1 else 4.0
            corners_2d = bbox_to_frustum_corners(det["bbox_2d"], det["distance_m"], length, P2)
        det["corners_2d"] = corners_2d

    annotated = annotate_image(scene["image"], detections, calib_parsed)
    bev = generate_bev(scene["points"], detections)

    return {
        "annotated_image":   annotated,
        "bev_image":         bev,
        "detections":        detections,
        "inference_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "num_points":        int(len(scene["points"])),
    }


# ── Public entry point ─────────────────────────────────────────────────────────

def process_zip(zip_bytes: bytes, max_frames: int = 20) -> dict:
    """
    Extract a KITTI ZIP, auto-detect format, run inference on up to max_frames
    frames, return a summary dict.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        cats = _categorise(names)

        # Sorted stems — process in dataset order
        common_stems = sorted(set(cats["bin"]) & set(cats["png"]))
        total_found = len(common_stems)
        stems_to_run = common_stems[:max_frames]

        frames = []
        errors = []
        for stem in stems_to_run:
            try:
                calib_dict = _build_calib_dict(zf, cats, stem)
                if calib_dict is None or "P2" not in calib_dict:
                    errors.append(f"{stem}: calib missing or incomplete")
                    continue

                bin_bytes = zf.read(cats["bin"][stem])
                img_bytes = zf.read(cats["png"][stem])

                result = _process_frame(bin_bytes, img_bytes, calib_dict)
                result["frame_id"] = stem
                frames.append(result)
            except Exception as exc:
                errors.append(f"{stem}: {exc}")

    return {
        "frames":         frames,
        "total_found":    total_found,
        "processed":      len(frames),
        "skipped_errors": errors,
    }
