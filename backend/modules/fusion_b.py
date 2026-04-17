"""
Enhanced frustum fusion — Approach B.

Improvements over Approach A (frustum median):
  1. RANSAC ground-plane removal  — eliminate road surface before depth estimation
  2. DBSCAN clustering            — isolate the target object, reject background noise
  3. PCA-based box fitting        — estimate orientation from point distribution
  4. Class-conditioned dim priors — blend with KITTI mean dims when cluster is sparse
"""
import numpy as np

from modules.calibration import project_points_to_image
from modules.fusion import _lidar_to_cam_xyz


# KITTI mean object dimensions (w, h, l) in metres ─────────────────────────
_DIM_PRIORS = {
    "car":        (1.62, 1.53, 3.88),
    "van":        (1.90, 2.05, 5.12),
    "truck":      (2.60, 3.07, 11.22),
    "pedestrian": (0.62, 1.76, 0.80),
    "person_sit": (0.62, 1.76, 0.80),
    "cyclist":    (0.60, 1.72, 1.76),
    "tram":       (2.50, 3.53, 16.10),
    "misc":       (1.80, 1.70, 4.50),
}
_DEFAULT_DIMS = (1.80, 1.70, 4.50)


def _remove_ground(points: np.ndarray) -> np.ndarray:
    """
    RANSAC-fit the ground plane to low-z points and remove everything on or
    below it.  Falls back to a fixed z-threshold if too few ground candidates.
    """
    if len(points) < 10:
        return points
    try:
        from sklearn.linear_model import RANSACRegressor, LinearRegression

        ground_cands = points[points[:, 2] < -0.4]
        if len(ground_cands) < 6:
            return points[points[:, 2] > -1.3]

        ransac = RANSACRegressor(
            LinearRegression(),
            residual_threshold=0.12,
            min_samples=max(6, int(0.5 * len(ground_cands))),
            max_trials=60,
        )
        ransac.fit(ground_cands[:, :2], ground_cands[:, 2])
        z_plane = ransac.predict(points[:, :2])
        return points[points[:, 2] > z_plane + 0.25]
    except Exception:
        return points[points[:, 2] > -1.2]


def _frustum_crop(
    projected: np.ndarray,
    points: np.ndarray,
    bbox: list,
    depth_min: float = 0.5,
    depth_max: float = 70.0,
) -> np.ndarray:
    """Crop points whose image projection falls inside bbox."""
    x1, y1, x2, y2 = bbox
    u, v, depth = projected[:, 0], projected[:, 1], projected[:, 2]
    mask = (
        (depth >= depth_min) & (depth <= depth_max)
        & (u >= x1) & (u <= x2)
        & (v >= y1) & (v <= y2)
    )
    return points[mask]


def _dbscan_largest(points: np.ndarray) -> np.ndarray:
    """Return the largest non-noise DBSCAN cluster (falls back to all points)."""
    if len(points) < 5:
        return points
    try:
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=0.6, min_samples=3, n_jobs=1).fit(points[:, :3])
        labels = db.labels_
        unique = [l for l in set(labels) if l != -1]
        if not unique:
            return points
        best = max(unique, key=lambda l: int((labels == l).sum()))
        return points[labels == best]
    except Exception:
        return points


def _pca_box(cam_pts: np.ndarray, cls: str) -> tuple:
    """
    Estimate oriented 3D box from camera-frame LiDAR cluster.

    Strategy:
      - Center Z (depth) from cluster median — most accurate LiDAR quantity
      - Center X, Y from cluster mean
      - L and W from KITTI class priors — frustum crops are unreliable for
        estimating length/width because they always span the full object depth
      - H blended: cluster vertical extent + prior (height is observable)
      - Yaw from PCA of horizontal footprint

    Returns (center [3], dims [w, h, l], yaw_rad).
    """
    prior_w, prior_h, prior_l = _DIM_PRIORS.get(cls.lower(), _DEFAULT_DIMS)

    if len(cam_pts) < 4:
        center = cam_pts.mean(axis=0) if len(cam_pts) else np.array([0.0, 1.0, 10.0])
        return center, np.array([prior_w, prior_h, prior_l]), 0.0

    # Depth (Z) — median of cluster; robust to outliers at frustum edges
    median_z = float(np.median(cam_pts[:, 2]))

    # Lateral center (X) — mean is fine; limited spread along X in frustum
    center_x = float(cam_pts[:, 0].mean())

    # Vertical center (Y, camera-down):
    #   LiDAR mostly illuminates the object's top surface (roof, bonnet).
    #   min(Y) ≈ top surface in camera frame; add h/2 to reach box center.
    top_y  = float(cam_pts[:, 1].min())
    raw_h  = float(cam_pts[:, 1].max() - cam_pts[:, 1].min())
    alpha  = float(np.clip((len(cam_pts) - 3) / 27.0, 0.0, 1.0))
    height = (alpha * raw_h + (1 - alpha) * prior_h) if raw_h > 0.15 else prior_h
    center_y = top_y + height / 2.0

    center = np.array([center_x, center_y, median_z])

    # Yaw: PCA on horizontal (XZ) footprint, then SNAP to nearest 90°.
    #
    # Raw PCA yaw is unreliable because frustum depth range always adds a
    # Z-elongation bias to the cluster, pulling the principal axis toward Z
    # regardless of the true object heading. Snapping to {0, ±π/2, π} gives
    # the four physically meaningful headings for street traffic (toward cam,
    # away, left-side-on, right-side-on) and eliminates diagonal artefacts.
    xz       = cam_pts[:, [0, 2]]
    centered = xz - xz.mean(axis=0)
    cov      = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    raw_yaw   = float(-np.arctan2(main_axis[0], main_axis[1]))
    # Snap to nearest multiple of π/2
    yaw = float(np.round(raw_yaw / (np.pi / 2)) * (np.pi / 2))

    # L and W always from class prior — frustum crops inflate extents
    return center, np.array([prior_w, height, prior_l]), yaw


def fuse_b(
    detections_2d: list[dict],
    points: np.ndarray,
    calib: dict,
    img_shape: tuple,
) -> list[dict]:
    """
    Approach B enhanced fusion pipeline.

    Steps per detection:
      1. Global ground removal (RANSAC)
      2. Frustum crop to 2D bbox
      3. DBSCAN → largest cluster
      4. PCA box fit with class priors
    """
    # Step 1: remove ground plane once for the whole scene
    points_clean = _remove_ground(points)

    projected = project_points_to_image(points_clean, calib, img_shape)

    result = []
    for det in detections_2d:
        x1, y1, x2, y2 = det["bbox_2d"]
        bbox_h = max(y2 - y1, 1)
        cls = det["class"]
        prior_w, prior_h, prior_l = _DIM_PRIORS.get(cls.lower(), _DEFAULT_DIMS)

        # Step 2: frustum crop
        cluster_lidar = _frustum_crop(projected, points_clean, det["bbox_2d"])

        if len(cluster_lidar) < 3:
            # Fallback: apparent-size heuristic + class prior dims
            est_dist = round(max(2.0, 500.0 / bbox_h), 2)
            xyz = [0.0, 1.0, est_dist]
            box_3d = [0.0, 1.0, est_dist, prior_w, prior_h, prior_l, 0.0]
        else:
            # Step 3: DBSCAN clustering
            cluster_lidar = _dbscan_largest(cluster_lidar)

            # Step 4: transform to camera frame, PCA box
            cam_pts = _lidar_to_cam_xyz(cluster_lidar, calib)
            center, dims, yaw = _pca_box(cam_pts, cls)

            xyz = [round(float(center[i]), 2) for i in range(3)]
            est_dist = max(float(center[2]), 0.1)
            box_3d = [
                round(float(center[0]), 3),
                round(float(center[1]), 3),
                round(float(center[2]), 3),
                round(float(dims[0]), 3),
                round(float(dims[1]), 3),
                round(float(dims[2]), 3),
                round(float(yaw), 4),
            ]

        result.append({
            "class":      cls,
            "confidence": det["confidence"],
            "bbox_2d":    det["bbox_2d"],
            "distance_m": round(est_dist, 2),
            "xyz":        xyz,
            "box_3d":     box_3d,
        })

    result.sort(key=lambda d: d["distance_m"])
    return result
