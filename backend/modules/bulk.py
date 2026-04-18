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
import os
import base64
import tempfile
import zipfile
from pathlib import PurePosixPath

import numpy as np
import cv2

from modules.loader import parse_calib_text, normalize_calib_dict, load_scene
from modules.calibration import parse_calib
from modules.fusion_pp import run_fused_pipeline
from modules.visualizer import render_lidar_bev_white, cv2_to_base64


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


def _build_calib_with_proj(calib_raw: dict) -> dict:
    """Build parsed calib plus T_velo_to_img, matching the notebook fusion path."""
    calib_raw = normalize_calib_dict(calib_raw)
    calib = parse_calib(calib_raw)
    calib["T_velo_to_img"] = calib["P2"] @ calib["R0_rect"] @ calib["Tr_velo_to_cam"]
    return calib


def _frame_sort_key(stem: str):
    """Sort KITTI frame stems numerically when possible, lexicographically otherwise."""
    return (0, int(stem)) if stem.isdigit() else (1, stem)


# ── Per-frame pipeline ─────────────────────────────────────────────────────────

def _process_frame(bin_bytes: bytes, img_bytes: bytes, calib_dict: dict) -> dict:
    import time
    t0 = time.perf_counter()

    scene = load_scene(bin_bytes, img_bytes, b"")
    calib = _build_calib_with_proj(calib_dict)

    image_bgr = cv2.cvtColor(scene["image"], cv2.COLOR_RGB2BGR)
    img_lidar, img_boxes, serial_dets, scene_points, scene_point_colors, stats = run_fused_pipeline(
        scene["points"], image_bgr, calib
    )
    lidar_bev = render_lidar_bev_white(scene["points"][:, :3], serial_dets)

    return {
        "camera_image":      cv2_to_base64(img_boxes),
        "lidar_image":       cv2_to_base64(img_lidar),
        "lidar_bev":         lidar_bev,
        "scene_points":      scene_points,
        "scene_point_colors": scene_point_colors,
        "detections":        serial_dets,
        "pipeline_stats":    stats,
        "inference_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "num_points":        int(len(scene["points"])),
    }


def _build_video_from_base64_frames(frames: list[dict], key: str, fps: float = 10.0) -> str | None:
    """Encode a list of base64 PNG frames into a base64 MP4 string."""
    if not frames:
        return None

    decoded_frames = []
    for frame in frames:
        b64 = frame.get(key)
        if not b64:
            continue
        try:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                decoded_frames.append(img)
        except Exception:
            continue

    if not decoded_frames:
        return None

    h, w = decoded_frames[0].shape[:2]
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            return None
        for img in decoded_frames:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(img)
        writer.release()

        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Public entry point ─────────────────────────────────────────────────────────

def process_zip(zip_bytes: bytes, max_frames: int = 20, is_timeseries: bool = True) -> dict:
    """
    Extract a KITTI ZIP, auto-detect format, run inference on up to max_frames
    frames, return a summary dict.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        cats = _categorise(names)

        # Follow KITTI frame ordering: iterate in camera timeline order, then filter
        # to valid frames with required files.
        camera_order = sorted(cats["png"].keys(), key=_frame_sort_key)

        # Process only frames that have all required per-frame files.
        # If per-frame calib exists in the ZIP, require it as well.
        if cats["calib_frame"]:
            common = set(cats["bin"]) & set(cats["png"]) & set(cats["calib_frame"])
        else:
            # Raw KITTI style uses date-level calib files, so stem-wise pair is bin+png.
            common = set(cats["bin"]) & set(cats["png"])

        common_stems = [s for s in camera_order if s in common]

        total_found = len(common_stems)
        stems_to_run = common_stems[:max_frames]

        frames = []
        errors = []
        for stem in stems_to_run:
            try:
                calib_dict = _build_calib_dict(zf, cats, stem)
                if calib_dict is None or "P2" not in calib_dict:
                    # Missing/incomplete calib for this frame: silently skip.
                    continue

                bin_bytes = zf.read(cats["bin"][stem])
                img_bytes = zf.read(cats["png"][stem])

                result = _process_frame(bin_bytes, img_bytes, calib_dict)
                result["frame_id"] = stem
                frames.append(result)
            except Exception as exc:
                errors.append(f"{stem}: {exc}")

    video_boxes_mp4 = _build_video_from_base64_frames(frames, "camera_image") if is_timeseries else None
    video_lidar_mp4 = _build_video_from_base64_frames(frames, "lidar_image") if is_timeseries else None

    return {
        "frames":         frames,
        "total_found":    total_found,
        "processed":      len(frames),
        "skipped_errors": errors,
        "is_timeseries":  bool(is_timeseries),
        "video_boxes_mp4": video_boxes_mp4,
        "video_lidar_mp4": video_lidar_mp4,
        "video_annotated_mp4": video_boxes_mp4,
        "video_bev_mp4":  video_lidar_mp4,
    }
