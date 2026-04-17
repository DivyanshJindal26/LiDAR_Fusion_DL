"""
KITTI scene loader — converts raw file bytes into numpy arrays.
"""
import io

import numpy as np
from PIL import Image


def parse_calib_text(text: str) -> dict:
    """
    Parse KITTI calibration file text.
    Returns dict keyed by field name (P2, R0_rect, Tr_velo_to_cam, etc.)
    with space-separated float strings as values.
    """
    calib = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        calib[key.strip()] = val.strip()
    return calib


def normalize_calib_dict(raw: dict) -> dict:
    """
    Accept a parsed calib dict from any KITTI format and return a dict with
    exactly the keys {P2, R0_rect, Tr_velo_to_cam} that parse_calib() expects.

    Handles three source formats:
      1. Object-detection per-frame calib.txt  — keys already correct
      2. KITTI cam-to-cam calib (P_rect_02, R_rect_00) merged with
         velo-to-cam calib (R 3×3, T 3×1 → assembled into 3×4 Tr string)
      3. Any mix of the above (e.g. user concatenated both files)
    """
    out = {}

    # P2
    if "P2" in raw:
        out["P2"] = raw["P2"]
    elif "P_rect_02" in raw:
        out["P2"] = raw["P_rect_02"]

    # R0_rect (always 3×3 stored as 9 floats)
    if "R0_rect" in raw:
        out["R0_rect"] = raw["R0_rect"]
    elif "R_rect_00" in raw:
        out["R0_rect"] = raw["R_rect_00"]

    # Tr_velo_to_cam: may already be present as 12-float 3×4 string,
    # or needs to be assembled from separate R (9 floats) + T (3 floats).
    if "Tr_velo_to_cam" in raw:
        out["Tr_velo_to_cam"] = raw["Tr_velo_to_cam"]
    elif "R" in raw and "T" in raw:
        try:
            R = [float(x) for x in raw["R"].split()]   # 9 values
            T = [float(x) for x in raw["T"].split()]   # 3 values
            # Build row-major 3×4: [r0 r1 r2 t0 | r3 r4 r5 t1 | r6 r7 r8 t2]
            tr = [R[0], R[1], R[2], T[0],
                  R[3], R[4], R[5], T[1],
                  R[6], R[7], R[8], T[2]]
            out["Tr_velo_to_cam"] = " ".join(f"{v:.10e}" for v in tr)
        except Exception:
            pass

    return out


def load_scene(bin_bytes: bytes, img_bytes: bytes, calib_bytes: bytes) -> dict:
    """
    Parse raw KITTI file bytes into usable numpy arrays.

    Returns:
        points: (N, 4) float32 — x, y, z, intensity
        image:  (H, W, 3) uint8 — RGB
        calib:  dict of raw string values keyed by KITTI field name
    """
    points = np.frombuffer(bin_bytes, dtype=np.float32).reshape(-1, 4)
    image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"), dtype=np.uint8)
    calib_text = calib_bytes.decode("utf-8", errors="replace")
    calib = parse_calib_text(calib_text)
    return {"points": points, "image": image, "calib": calib}
