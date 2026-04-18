"""
Full fused pipeline from PPimproved2.ipynb — ported verbatim to operate on
numpy arrays instead of file paths.

Pipeline per frame:
  1. Ground removal + forward filter
  2. YOLOv8l 2D detection (conf ≥ 0.3, classes of interest)
  3. PointPillars 3D detection (score ≥ 0.4)
  4. Gate PP with YOLO (IoU ≥ 0.25, class-aware, score ≥ 0.45)
  5. DBSCAN+OBB pipeline for each YOLO box
  6. Class-aware merge of PP and OBB detections
  7. Render img_lidar (LiDAR depth dots) and img_boxes (3D wireframes)

Returns: img_lidar, img_boxes, final_dets, stats
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from modules.pointpillars import run_pointpillars

# ── YOLO (reuse existing loader) ─────────────────────────────────────────────
_yolo_model = None

def _load_yolo():
    global _yolo_model
    try:
        from ultralytics import YOLO
        print("[fusion_pp] loading YOLOv8l...", flush=True)
        _yolo_model = YOLO("yolov8l.pt")
        print("[fusion_pp] YOLOv8l ready", flush=True)
    except Exception as exc:
        print(f"[fusion_pp] YOLO failed: {exc}", flush=True)

_load_yolo()


# ── Constants (verbatim from notebook) ───────────────────────────────────────

CLASSES_OF_INTEREST = {
    "car", "truck", "bus", "person", "pedestrian",
    "bicycle", "cyclist", "motorcycle",
}

CLASS_PRIORS = {
    "car":        np.array([4.5, 2.0, 1.7]),
    "truck":      np.array([7.0, 2.5, 3.0]),
    "bus":        np.array([10.0, 2.5, 3.5]),
    "person":     np.array([0.6, 0.6, 1.8]),
    "pedestrian": np.array([0.6, 0.6, 1.8]),
    "cyclist":    np.array([1.8, 0.6, 1.7]),
    "bicycle":    np.array([1.8, 0.6, 1.0]),
}

YOLO_TO_PP_CLASS = {
    "car":        ["car", "vehicle"],
    "truck":      ["truck", "vehicle"],
    "bus":        ["car", "truck"],
    "person":     ["pedestrian"],
    "pedestrian": ["pedestrian"],
    "cyclist":    ["cyclist"],
    "bicycle":    ["cyclist"],
}

# BGR wireframe colors per confidence tier (matches notebook)
TIER_COLORS = {
    "HIGH (PP+OBB agree)":        (0, 255, 0),
    "HIGH (PP only, YOLO confirmed)": (0, 200, 100),
    "MED (OBB only — sparse LiDAR)": (0, 165, 255),
}

# Frontend display colors (hex) per class
CLASS_HEX = {
    "car":        "#2979ff",
    "pedestrian": "#00e676",
    "cyclist":    "#ffab00",
    "truck":      "#ff3d71",
    "bus":        "#e040fb",
    "person":     "#00e676",
    "bicycle":    "#ffab00",
    "motorcycle": "#ffab00",
}


# ── Notebook helper functions (verbatim) ──────────────────────────────────────

def box_lwh_center_to_corners(center, l, w, h, yaw):
    ca, sa = np.cos(yaw), np.sin(yaw)
    R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    dx, dy, dz = l / 2, w / 2, h / 2
    local = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx,  dy, -dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [dx, -dy,  dz], [dx,  dy,  dz], [-dx,  dy,  dz],
    ])
    return ((R @ local.T).T + center).astype(np.float32)


def iou_2d(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-6)


def project_pts_to_image(points_xyz, calib, img_shape):
    h, w = img_shape[:2]
    N = points_xyz.shape[0]
    pts_h = np.vstack([points_xyz.T, np.ones((1, N))])
    proj  = calib["T_velo_to_img"] @ pts_h
    depth = proj[2]
    u = proj[0] / (depth + 1e-9)
    v = proj[1] / (depth + 1e-9)
    valid = (depth > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[valid], v[valid], depth[valid], valid


def get_frustum_points(box2d, pts_xyz, u_all, v_all, valid_mask):
    x1, y1, x2, y2 = box2d
    in_box  = (u_all >= x1) & (u_all <= x2) & (v_all >= y1) & (v_all <= y2)
    indices = np.where(valid_mask)[0][in_box]
    return pts_xyz[indices]


def dbscan_cluster(pts):
    if len(pts) < 5:
        return None
    db     = DBSCAN(eps=0.5, min_samples=3).fit(pts)
    labels = db.labels_
    unique = [l for l in set(labels) if l != -1]
    if not unique:
        return None
    best = max(unique, key=lambda l: np.sum(labels == l))
    return pts[labels == best]


def fit_obb(pts):
    if pts is None or len(pts) < 5:
        return None
    xy = pts[:, :2]
    z  = pts[:, 2]

    pca   = PCA(n_components=2).fit(xy)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    c, s  = np.cos(-angle), np.sin(-angle)
    R     = np.array([[c, -s], [s, c]])
    xy_r  = (R @ xy.T).T

    xmin, ymin = xy_r.min(0); xmax, ymax = xy_r.max(0)
    zmin, zmax = z.min(), z.max()

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    center_xy = np.array([R.T @ [cx, cy]])[0]
    center    = np.array([center_xy[0], center_xy[1], (zmin + zmax) / 2])
    dims      = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

    corners_r = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ])
    corners = np.zeros_like(corners_r)
    for i in range(8):
        corners[i, :2] = R.T @ corners_r[i, :2]
        corners[i, 2]  = corners_r[i, 2]

    return {"center": center, "dims": dims, "corners": corners,
            "angle": angle, "source": "obb"}


def fit_prior_box(pts, cls):
    if pts is None or len(pts) == 0:
        return None
    center = pts.mean(axis=0)
    dims   = CLASS_PRIORS.get(cls, CLASS_PRIORS["car"])
    l, w, h = dims
    corners = np.array([
        [center[0]-l/2, center[1]-w/2, center[2]],
        [center[0]+l/2, center[1]-w/2, center[2]],
        [center[0]+l/2, center[1]+w/2, center[2]],
        [center[0]-l/2, center[1]+w/2, center[2]],
        [center[0]-l/2, center[1]-w/2, center[2]+h],
        [center[0]+l/2, center[1]-w/2, center[2]+h],
        [center[0]+l/2, center[1]+w/2, center[2]+h],
        [center[0]-l/2, center[1]+w/2, center[2]+h],
    ])
    return {"center": center, "dims": dims, "corners": corners,
            "angle": 0.0, "source": "prior"}


def run_old_pipeline(yolo_boxes, pts_xyz, calib, img_shape):
    u_all, v_all, depth_all, valid_mask = project_pts_to_image(pts_xyz, calib, img_shape)
    dets = []
    for yb in yolo_boxes:
        x1, y1, x2, y2 = yb["x1"], yb["y1"], yb["x2"], yb["y2"]
        cls  = yb["class"]
        conf = yb["conf"]

        fpts    = get_frustum_points((x1, y1, x2, y2), pts_xyz, u_all, v_all, valid_mask)
        cluster = dbscan_cluster(fpts)

        if cluster is not None and len(cluster) >= 10:
            box = fit_obb(cluster)
        else:
            box = fit_prior_box(
                fpts if fpts is not None and len(fpts) > 0 else None, cls
            )

        if box is None:
            continue

        dets.append({
            "label":   cls,
            "score":   conf,
            "center":  box["center"],
            "dims":    box["dims"],
            "corners": box["corners"],
            "angle":   box["angle"],
            "bbox_2d": (x1, y1, x2, y2),
            "source":  box["source"],
        })
    return dets


# ── Inferred helpers (not in visible notebook cells) ──────────────────────────

def project_box_corners_to_image(corners_3d, calib, img_shape):
    """Project 8 LiDAR-frame 3D corners through T_velo_to_img; return 2D bbox or None."""
    h, w = img_shape[:2]
    N    = 8
    pts_h = np.vstack([corners_3d.T, np.ones((1, N))])
    proj  = calib["T_velo_to_img"] @ pts_h
    depth = proj[2]
    if np.all(depth <= 0):
        return None
    valid = depth > 0
    u = proj[0][valid] / (depth[valid] + 1e-9)
    v = proj[1][valid] / (depth[valid] + 1e-9)
    x1, y1 = max(0, int(u.min())), max(0, int(v.min()))
    x2, y2 = min(w, int(u.max())), min(h, int(v.max()))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def is_valid_box(det):
    l, w, h = det["dims"]
    return (0.3 < l < 20) and (0.3 < w < 8) and (0.3 < h < 6)


def box_iou_3d_centers(det_a, det_b, dist_thresh):
    return float(np.linalg.norm(det_a["center"] - det_b["center"])) < dist_thresh


# ── PP gating and merge (verbatim from notebook) ──────────────────────────────

def gate_pp_with_yolo(pp_dets, yolo_boxes, calib, img_shape,
                      iou_thresh=0.25, score_thresh=0.45):
    gated = []
    for det in pp_dets:
        if det["score"] < score_thresh:
            continue
        if not is_valid_box(det):
            continue

        pp_rect = project_box_corners_to_image(det["corners"], calib, img_shape)
        if pp_rect is None:
            continue

        matched = False
        for yb in yolo_boxes:
            yolo_rect = (yb["x1"], yb["y1"], yb["x2"], yb["y2"])
            iou = iou_2d(pp_rect, yolo_rect)
            if iou >= iou_thresh:
                pp_cls   = det["label"].lower()
                yolo_cls = yb["class"].lower()
                allowed  = YOLO_TO_PP_CLASS.get(yolo_cls, [yolo_cls])
                if pp_cls in allowed:
                    matched = True
                    det["label"] = yolo_cls
                    break

        if matched:
            det["bbox_2d"] = pp_rect
            det["source"]  = "pp_gated"
            gated.append(det)
    return gated


def merge_detections(pp_dets, old_dets, dist_thresh=3.0):
    final    = []
    used_old = set()

    for pp in pp_dets:
        best_match = None
        for i, old in enumerate(old_dets):
            if i in used_old:
                continue
            if box_iou_3d_centers(pp, old, dist_thresh) and pp["label"] == old["label"]:
                best_match = i
                break

        if best_match is not None:
            pp["confidence_tier"] = "HIGH (PP+OBB agree)"
            pp["color"] = (0, 255, 0)
            final.append(pp)
            used_old.add(best_match)
        else:
            pp["confidence_tier"] = "HIGH (PP only, YOLO confirmed)"
            pp["color"] = (0, 200, 100)
            final.append(pp)

    for i, old in enumerate(old_dets):
        if i in used_old:
            continue
        old["confidence_tier"] = "MED (OBB only — sparse LiDAR)"
        old["color"] = (0, 165, 255)
        final.append(old)

    return final


# ── Public entry point ────────────────────────────────────────────────────────

def run_fused_pipeline(
    pts_raw: np.ndarray,
    image_bgr: np.ndarray,
    calib: dict,
    score_thresh: float = 0.4,
) -> tuple:
    """
    Verbatim adaptation of notebook run_fused_pipeline.
    Takes numpy arrays instead of file paths.

    calib must contain 'T_velo_to_img' = P2 @ R0_rect @ Tr_velo_to_cam.

    Returns:
        img_lidar   (H,W,3) uint8 BGR — camera image + LiDAR depth dots
        img_boxes   (H,W,3) uint8 BGR — camera image + 3D wireframes + labels
        final_dets  list of detection dicts (serialisable — corners as lists)
        stats       dict {yolo_n, pp_raw_n, pp_gated_n, obb_n, final_n}
    """
    # ── Preprocess LiDAR ─────────────────────────────────────────────────────
    pts_xyz = pts_raw[:, :3].astype(np.float32)
    pts_xyz = pts_xyz[pts_xyz[:, 2] > -1.5]
    pts_xyz = pts_xyz[pts_xyz[:, 0] > 0]

    img_shape = image_bgr.shape

    # ── YOLO 2D ──────────────────────────────────────────────────────────────
    yolo_boxes = []
    if _yolo_model is not None:
        try:
            yolo_results = _yolo_model(image_bgr, verbose=False)[0]
            for box in yolo_results.boxes:
                cls  = yolo_results.names[int(box.cls)]
                conf = float(box.conf)
                if cls not in CLASSES_OF_INTEREST or conf < 0.3:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                yolo_boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "class": cls, "conf": conf,
                })
        except Exception as exc:
            print(f"[fusion_pp] YOLO error: {exc}", flush=True)

    # ── PointPillars ─────────────────────────────────────────────────────────
    pp_raw  = run_pointpillars(pts_xyz, score_thresh=0.4)
    pp_dets = gate_pp_with_yolo(pp_raw, yolo_boxes, calib, img_shape,
                                iou_thresh=0.25, score_thresh=0.45)

    # ── DBSCAN + OBB ─────────────────────────────────────────────────────────
    old_dets = run_old_pipeline(yolo_boxes, pts_xyz, calib, img_shape)

    # ── Merge ─────────────────────────────────────────────────────────────────
    final_dets = merge_detections(pp_dets, old_dets, dist_thresh=3.0)

    stats = {
        "yolo_n":    len(yolo_boxes),
        "pp_raw_n":  len(pp_raw),
        "pp_gated_n": len(pp_dets),
        "obb_n":     len(old_dets),
        "final_n":   len(final_dets),
    }
    print(f"[fusion_pp] YOLO:{stats['yolo_n']}  PP_raw:{stats['pp_raw_n']}  "
          f"PP_gated:{stats['pp_gated_n']}  OBB:{stats['obb_n']}  "
          f"Final:{stats['final_n']}", flush=True)

    # ── Draw img_lidar (verbatim notebook) ────────────────────────────────────
    img_lidar = image_bgr.copy()
    u_all, v_all, d_all, vmask = project_pts_to_image(pts_xyz, calib, img_shape)
    if len(d_all) > 0:
        d_norm = (d_all - d_all.min()) / (d_all.max() - d_all.min() + 1e-9)
        for i in range(len(u_all)):
            c = plt.cm.jet(float(d_norm[i]))
            cv2.circle(
                img_lidar,
                (int(u_all[i]), int(v_all[i])),
                2,
                (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)),
                -1,
            )

    # ── Draw img_boxes (verbatim notebook) ────────────────────────────────────
    img_boxes = image_bgr.copy()
    for det in final_dets:
        color = det.get("color", (0, 255, 0))

        c2d = project_box_corners_to_image(det["corners"], calib, img_shape)
        if c2d:
            N     = 8
            pts_h = np.vstack([det["corners"].T, np.ones((1, N))])
            proj  = calib["T_velo_to_img"] @ pts_h
            dep   = proj[2]
            uu = (proj[0] / (dep + 1e-9)).astype(int)
            vv = (proj[1] / (dep + 1e-9)).astype(int)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for a, b in edges:
                p1, p2 = (uu[a], vv[a]), (uu[b], vv[b])
                if all(-500 < x < 3000 for x in p1 + p2):
                    cv2.line(img_boxes, p1, p2, color, 2)

        dist = float(np.linalg.norm(det["center"]))
        src  = det.get("source", "")
        x1, y1, x2, y2 = det.get("bbox_2d", (0, 0, 0, 0))
        cv2.rectangle(img_boxes, (x1, y1), (x2, y2), color, 1)
        label_txt = f"{det['label']} {dist:.1f}m [{src}]"
        cv2.putText(img_boxes, label_txt, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ── Serialise detections for JSON response ────────────────────────────────
    serial_dets = []
    for det in final_dets:
        serial_dets.append({
            "label":           det["label"],
            "score":           round(float(det["score"]), 3),
            "center":          [round(float(v), 3) for v in det["center"]],
            "dims":            [round(float(v), 3) for v in det["dims"]],
            "corners":         [[round(float(v), 3) for v in row]
                                for row in det["corners"].tolist()],
            "heading":         round(float(det.get("heading", det.get("angle", 0.0))), 4),
            "bbox_2d":         list(det.get("bbox_2d", [0, 0, 0, 0])),
            "source":          det.get("source", ""),
            "confidence_tier": det.get("confidence_tier", ""),
            "color_hex":       CLASS_HEX.get(det["label"].lower(), "#94a3b8"),
            "distance_m":      round(float(np.linalg.norm(det["center"])), 2),
        })

    return img_lidar, img_boxes, serial_dets, stats
