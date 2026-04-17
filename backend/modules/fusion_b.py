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


def _dbscan_depth_seeded(points: np.ndarray, target_x: float) -> np.ndarray:
    """
    Run DBSCAN and return the cluster whose median forward distance
    (LiDAR X = forward) is closest to `target_x`.

    This replaces the old "largest cluster" strategy. For distant objects the
    frustum sweeps through dense foreground at shorter range — largest-cluster
    picks the foreground (e.g. 25m) instead of the actual target (e.g. 58m).
    Seeding with the apparent-size depth estimate avoids this.
    """
    if len(points) < 5:
        return points
    try:
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=0.6, min_samples=3, n_jobs=1).fit(points[:, :3])
        labels = db.labels_
        unique = [l for l in set(labels) if l != -1]
        if not unique:
            return points
        best = min(unique,
                   key=lambda l: abs(float(np.median(points[labels == l, 0])) - target_x))
        return points[labels == best]
    except Exception:
        return points


def _pca_box(cam_pts: np.ndarray, cls: str) -> tuple:
    """
    Fit a tight oriented 3D bounding box (OBB) to a camera-frame LiDAR cluster.

    Dimension strategy:
      - Z (depth): median of cluster — most accurate LiDAR quantity
      - X (lateral): mean — limited spread inside frustum
      - Y (vertical): min(cluster Y) + H/2 — LiDAR hits top surfaces; shift down
      - H: cluster vertical extent blended with class prior
      - W, L: PCA principal-axis extents — actual measured size from cluster.
              Floor-clamped to 50% of class prior so a single-face crop (e.g.
              head-on car where only the front face is lit) can't collapse to zero.
      - Yaw: raw PCA angle of the major horizontal axis — no 90° snap.
             The snap was cosmetically clean but wrong for diagonal/merging objects.
    """
    prior_w, prior_h, prior_l = _DIM_PRIORS.get(cls.lower(), _DEFAULT_DIMS)

    if len(cam_pts) < 4:
        center = cam_pts.mean(axis=0) if len(cam_pts) else np.array([0.0, 1.0, 10.0])
        return center, np.array([prior_w, prior_h, prior_l]), 0.0

    # ── Depth (Z) ────────────────────────────────────────────────────────────
    median_z = float(np.median(cam_pts[:, 2]))
    center_x = float(cam_pts[:, 0].mean())

    # ── Height (Y, camera-down) ───────────────────────────────────────────────
    top_y  = float(cam_pts[:, 1].min())
    raw_h  = float(cam_pts[:, 1].max() - cam_pts[:, 1].min())
    alpha  = float(np.clip((len(cam_pts) - 3) / 27.0, 0.0, 1.0))
    height = (alpha * raw_h + (1 - alpha) * prior_h) if raw_h > 0.15 else prior_h
    center_y = top_y + height / 2.0

    center = np.array([center_x, center_y, median_z])

    # ── Oriented footprint (XZ plane) via PCA ────────────────────────────────
    xz       = cam_pts[:, [0, 2]]
    centered = xz - xz.mean(axis=0)
    cov      = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Major axis = length direction; minor axis = width direction
    major_idx  = np.argmax(eigenvalues)
    minor_idx  = 1 - major_idx
    main_axis  = eigenvectors[:, major_idx]
    minor_axis = eigenvectors[:, minor_idx]

    # Project cluster onto each axis → measure actual extents
    proj_l = centered @ main_axis
    proj_w = centered @ minor_axis
    meas_l = float(proj_l.max() - proj_l.min())
    meas_w = float(proj_w.max() - proj_w.min())

    # Floor at 50% of class prior — single visible face gives underestimate
    length = max(meas_l, prior_l * 0.50)
    width  = max(meas_w, prior_w * 0.50)

    # Raw PCA yaw — accurate for diagonal / lane-changing objects
    yaw = float(-np.arctan2(main_axis[0], main_axis[1]))

    return center, np.array([width, height, length]), yaw


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

    # Extract fy from P2 for apparent-size depth estimation
    P2 = calib.get("P2")
    fy = float(P2[1, 1]) if P2 is not None else 721.5

    result = []
    for det in detections_2d:
        x1, y1, x2, y2 = det["bbox_2d"]
        bbox_h = max(y2 - y1, 1)
        cls = det["class"]
        prior_w, prior_h, prior_l = _DIM_PRIORS.get(cls.lower(), _DEFAULT_DIMS)

        # Apparent-size depth seed: fy * height / bbox_height_px
        # Universal height constant avoids wrong seeds when the detected class
        # doesn't match the actual object (e.g. YOLO says Truck but GT is Misc).
        # 1.65m ≈ average of all KITTI road objects (cars, pedestrians, cyclists, vans).
        z_apparent = fy * 1.65 / bbox_h

        # Step 2: frustum crop
        cluster_lidar = _frustum_crop(projected, points_clean, det["bbox_2d"])

        if len(cluster_lidar) < 3:
            # Fallback: apparent-size heuristic + class prior dims
            est_dist = round(max(2.0, z_apparent), 2)
            xyz = [0.0, 1.0, est_dist]
            box_3d = [0.0, 1.0, est_dist, prior_w, prior_h, prior_l, 0.0]
        else:
            # Step 3: depth-seeded DBSCAN — pick cluster nearest to apparent depth
            cluster_lidar = _dbscan_depth_seeded(cluster_lidar, z_apparent)

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
