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
    """
    Encode a list of base64 PNG frames into a browser-playable H.264 MP4 (base64).
    Strategy: write frames as JPEGs to a temp dir, then encode with ffmpeg (H.264).
    Falls back to cv2 avc1 → mp4v if ffmpeg is not available.
    """
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
    # Ensure even dimensions — H.264 requires width/height divisible by 2
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "out.mp4")
    try:
        # Write frames as numbered JPEGs
        for i, img in enumerate(decoded_frames):
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(tmp_dir, f"frame_{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Try ffmpeg first — produces browser-native H.264
        import subprocess
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%06d.jpg"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path,
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")

        # ffmpeg failed — fall back to cv2 avc1 then mp4v
        print(f"[bulk] ffmpeg failed: {result.stderr.decode()[:200]}", flush=True)
        for fourcc_str in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h)
            )
            if writer.isOpened():
                for img in decoded_frames:
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                    writer.write(img)
                writer.release()
                if os.path.getsize(out_path) > 0:
                    with open(out_path, "rb") as f:
                        return base64.b64encode(f.read()).decode("ascii")
            writer.release()

        return None

    except Exception as exc:
        print(f"[bulk] video encode error: {exc}", flush=True)
        return None
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Streaming entry point ──────────────────────────────────────────────────────

def stream_process_zip(zip_bytes: bytes, is_timeseries: bool = True):
    """
    Generator that yields SSE-ready dicts:
      {"type": "start",    "total": N}
      {"type": "progress", "current": i, "total": N, "frame_id": stem}  — one per frame
      {"type": "encoding"}                                                 — video encoding phase
      {"type": "done",     "frames": [...], "video_annotated_mp4": ..., ...}
    Errors on individual frames are reported inside the progress event as "error".
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        cats  = _categorise(names)

        camera_order = sorted(cats["png"].keys(), key=_frame_sort_key)

        if cats["calib_frame"]:
            common = set(cats["bin"]) & set(cats["png"]) & set(cats["calib_frame"])
        else:
            common = set(cats["bin"]) & set(cats["png"])

        common_stems = [s for s in camera_order if s in common]
        total_found  = len(common_stems)

        yield {"type": "start", "total": total_found}

        frames = []
        errors = []
        for i, stem in enumerate(common_stems):
            try:
                calib_dict = _build_calib_dict(zf, cats, stem)
                if calib_dict is None or "P2" not in calib_dict:
                    errors.append(f"{stem}: missing calib")
                    yield {"type": "progress", "current": i + 1, "total": total_found,
                           "frame_id": stem, "error": "missing calib"}
                    continue

                bin_b = zf.read(cats["bin"][stem])
                img_b = zf.read(cats["png"][stem])
                result = _process_frame(bin_b, img_b, calib_dict)
                result["frame_id"] = stem
                frames.append(result)
                yield {"type": "progress", "current": i + 1, "total": total_found, "frame_id": stem}
            except Exception as exc:
                errors.append(f"{stem}: {exc}")
                yield {"type": "progress", "current": i + 1, "total": total_found,
                       "frame_id": stem, "error": str(exc)}

    # Video encoding phase (can take a while — signal the frontend)
    if is_timeseries and frames:
        yield {"type": "encoding"}

    video_annotated = _build_video_from_base64_frames(frames, "camera_image") if is_timeseries else None
    video_lidar     = _build_video_from_base64_frames(frames, "lidar_image")  if is_timeseries else None
    video_bev       = _build_video_from_base64_frames(frames, "lidar_bev")    if is_timeseries else None

    yield {
        "type":                "done",
        "frames":              frames,
        "total_found":         total_found,
        "processed":           len(frames),
        "skipped_errors":      errors,
        "is_timeseries":       bool(is_timeseries),
        "video_annotated_mp4": video_annotated,
        "video_lidar_mp4":     video_lidar,
        "video_bev_mp4":       video_bev,
    }
