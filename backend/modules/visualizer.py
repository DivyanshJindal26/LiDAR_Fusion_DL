"""
Visualization: annotated camera image + bird's-eye-view plot, both as base64 PNG.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw

# Distance color stops — matches frontend colorScale.js exactly
_STOPS = [
    (0,  (239, 68,  68)),
    (5,  (249, 115, 22)),
    (10, (245, 158, 11)),
    (20, (132, 204, 22)),
    (35, (34,  197, 94)),
]

CLASS_COLORS_PIL = {
    "Car":        (96,  165, 250),   # blue
    "Pedestrian": (52,  211, 153),   # green
    "Cyclist":    (251, 191, 36),    # amber
    "Van":        (167, 139, 250),   # purple
    "Truck":      (244, 114, 182),   # pink
}

# Edge convention: corners 0-3 = near face, 4-7 = far face
# 0=right-bottom-near  1=left-bottom-near  2=left-top-near   3=right-top-near
# 4=right-bottom-far   5=left-bottom-far   6=left-top-far    7=right-top-far
FRONT_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
BACK_EDGES  = [(4, 5), (5, 6), (6, 7), (7, 4)]
SIDE_EDGES  = [(0, 4), (1, 5), (2, 6), (3, 7)]


def _distance_to_rgb(dist_m: float) -> tuple:
    d = max(0.0, float(dist_m))
    for i in range(len(_STOPS) - 1):
        lo_d, lo_c = _STOPS[i]
        hi_d, hi_c = _STOPS[i + 1]
        if d <= hi_d:
            t = (d - lo_d) / (hi_d - lo_d)
            return tuple(int(lo_c[j] + (hi_c[j] - lo_c[j]) * t) for j in range(3))
    return _STOPS[-1][1]


def _class_color(cls: str) -> tuple:
    return CLASS_COLORS_PIL.get(cls, (148, 163, 184))


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="#020617")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Frustum-based 3D corner computation ───────────────────────────────────────

def bbox_to_frustum_corners(bbox_2d: list, distance_m: float, length: float,
                             P2: np.ndarray) -> list:
    """
    Build the 8 image-space corners of a 3D box using the 2D detection bbox
    as the near face and perspective projection for the far face.

    This always produces a clean, correctly-proportioned box in image space
    regardless of yaw estimation errors, because the near face is anchored
    directly to the 2D detection.

    Corner ordering (matches FRONT/BACK/SIDE_EDGES above):
        0: right-bottom-near   1: left-bottom-near
        2: left-top-near       3: right-top-near
        4: right-bottom-far    5: left-bottom-far
        6: left-top-far        7: right-top-far

    Returns list of 8 [u, v] pairs.
    """
    x1, y1, x2, y2 = [float(c) for c in bbox_2d]
    cx_pix = float(P2[0, 2])
    cy_pix = float(P2[1, 2])

    # Near face: front surface of the object (closer to camera)
    z_near = max(distance_m - length / 2.0, distance_m * 0.55, 0.5)
    # Far face: back surface
    z_far  = distance_m + length / 2.0

    # Perspective scale: far corners converge toward the principal point
    scale = z_near / z_far

    def far_pt(u, v):
        return [cx_pix + (u - cx_pix) * scale,
                cy_pix + (v - cy_pix) * scale]

    near = [
        [x2, y2],  # 0 right-bottom
        [x1, y2],  # 1 left-bottom
        [x1, y1],  # 2 left-top
        [x2, y1],  # 3 right-top
    ]
    far = [
        far_pt(x2, y2),  # 4
        far_pt(x1, y2),  # 5
        far_pt(x1, y1),  # 6
        far_pt(x2, y1),  # 7
    ]
    return near + far


# Keep old box3d projection for BEV / export consumers ────────────────────────

def box3d_corners_cam(box_3d: list) -> np.ndarray:
    cx, cy, cz, w, h, l, yaw = box_3d
    hw, hh, hl = w / 2.0, h / 2.0, l / 2.0
    local = np.array([
        [ hw,  hh,  hl], [-hw,  hh,  hl], [-hw, -hh,  hl], [ hw, -hh,  hl],
        [ hw,  hh, -hl], [-hw,  hh, -hl], [-hw, -hh, -hl], [ hw, -hh, -hl],
    ], dtype=np.float64)
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], dtype=np.float64)
    return local @ R.T + np.array([cx, cy, cz], dtype=np.float64)


def project_box3d(box_3d: list, P2: np.ndarray) -> np.ndarray | None:
    corners = box3d_corners_cam(box_3d)
    hom  = np.hstack([corners, np.ones((8, 1))])
    proj = P2 @ hom.T
    d    = proj[2]
    if (d <= 0.1).any():
        return None
    return np.stack([proj[0] / d, proj[1] / d], axis=1)


# ── Image annotation ──────────────────────────────────────────────────────────

def _draw_box3d_pil(draw: ImageDraw.Draw, corners: list, color: tuple,
                    front_w: int = 3, back_w: int = 1) -> None:
    """Draw the 12-edge wireframe from 8 [u,v] corner pairs."""
    def pt(i):
        return (int(round(corners[i][0])), int(round(corners[i][1])))

    # Near (front) face — thick, bright
    for i, j in FRONT_EDGES:
        draw.line([pt(i), pt(j)], fill=color, width=front_w)
    # Far (back) face — thin
    dim = tuple(max(0, c - 60) for c in color)
    for i, j in BACK_EDGES:
        draw.line([pt(i), pt(j)], fill=dim, width=back_w)
    # Pillars
    for i, j in SIDE_EDGES:
        draw.line([pt(i), pt(j)], fill=dim, width=back_w)


def annotate_image(image: np.ndarray, detections: list[dict], calib: dict,
                   ground_truth: list[dict] | None = None) -> str:
    """
    Draw frustum 3D bounding boxes on the camera image.
    Near face = 2D detection bbox (always fits cleanly).
    Far face  = perspective-scaled toward vanishing point.
    Returns base64 PNG.
    """
    P2 = calib.get("P2")
    img  = Image.fromarray(image.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)

    # GT boxes — dashed white 2D outlines
    if ground_truth:
        for gt in ground_truth:
            x1, y1, x2, y2 = gt["bbox_2d"]
            dash, gap = 8, 4
            for x in range(x1, x2, dash + gap):
                draw.line([(x, y1), (min(x + dash, x2), y1)], fill=(255, 255, 255), width=2)
                draw.line([(x, y2), (min(x + dash, x2), y2)], fill=(255, 255, 255), width=2)
            for y in range(y1, y2, dash + gap):
                draw.line([(x1, y), (x1, min(y + dash, y2))], fill=(255, 255, 255), width=2)
                draw.line([(x2, y), (x2, min(y + dash, y2))], fill=(255, 255, 255), width=2)
            lbl = f"GT:{gt['class']} {gt.get('distance_m', 0):.1f}m"
            tw  = draw.textlength(lbl)
            draw.rectangle([x1, y2 + 1, x1 + tw + 6, y2 + 16], fill=(60, 60, 60))
            draw.text((x1 + 3, y2 + 2), lbl, fill=(220, 220, 220))

    # Prediction 3D wireframes
    for det in detections:
        dist   = det.get("distance_m", 0.0)
        color  = _distance_to_rgb(dist)
        box3d  = det.get("box_3d")
        bbox2d = det.get("bbox_2d")

        if P2 is not None and box3d and bbox2d:
            length  = float(box3d[5]) if box3d[5] > 0.1 else 4.0
            corners = bbox_to_frustum_corners(bbox2d, dist, length, P2)
            _draw_box3d_pil(draw, corners, color, front_w=3, back_w=1)
            lx = int(min(c[0] for c in corners[:4]))
            ly = int(min(c[1] for c in corners[:4])) - 18
        elif bbox2d:
            x1, y1, x2, y2 = bbox2d
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            lx, ly = x1, max(y1 - 18, 0)
        else:
            continue

        label = f"{det['class']} {dist:.1f}m"
        tw    = draw.textlength(label)
        ly    = max(ly, 0)
        draw.rectangle([lx, ly, lx + tw + 6, ly + 15], fill=(*color, 220))
        draw.text((lx + 3, ly + 2), label, fill=(10, 10, 10))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── BEV ───────────────────────────────────────────────────────────────────────

def generate_bev(points: np.ndarray, detections: list[dict]) -> str:
    """
    Dense top-down point cloud colored by height + oriented box footprints.
    Returns base64 PNG.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#020617")
    ax.set_facecolor("#020617")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-5, 50)
    ax.set_aspect("equal")

    # ── Point cloud colored by height ──────────────────────────────────────
    n_show = min(len(points), 15_000)
    idx    = rng.choice(len(points), n_show, replace=False)
    sub    = points[idx]
    # BEV axes: bev_x = -LiDAR_Y (right), bev_y = LiDAR_X (forward)
    bev_x = -sub[:, 1]
    bev_y =  sub[:, 0]
    # Color by height (LiDAR Z), clipped to [-2, 2] m
    z_norm = np.clip((sub[:, 2] + 2.0) / 4.0, 0.0, 1.0)
    ax.scatter(bev_x, bev_y, c=z_norm, cmap="plasma",
               s=0.5, alpha=0.5, linewidths=0, vmin=0, vmax=1)

    # ── Distance rings ─────────────────────────────────────────────────────
    ring_data = [(10, "#ef4444"), (20, "#f97316"), (30, "#84cc16"), (40, "#22c55e")]
    theta = np.linspace(0, 2 * np.pi, 360)
    for r, c in ring_data:
        ax.plot(r * np.sin(theta), r * np.cos(theta), color=c, lw=0.6, ls="--", alpha=0.4)
        ax.text(0.5, r + 0.5, f"{r}m", color=c, fontsize=6, alpha=0.7,
                ha="center", fontfamily="monospace")

    # ── Ego vehicle ────────────────────────────────────────────────────────
    ax.scatter([0], [0], marker="^", s=80, color="#3b82f6",
               edgecolors="#93c5fd", linewidths=1.5, zorder=6)

    # ── Detection footprints ───────────────────────────────────────────────
    for det in detections:
        cls  = det["class"]
        col  = [c / 255 for c in _class_color(cls)]
        box  = det.get("box_3d")
        xyz  = det.get("xyz", [0, 0, 0])
        # BEV center: cam_x → bev_x, cam_z → bev_y
        bx, bz = float(xyz[0]), float(xyz[2])

        if box and len(box) == 7:
            _, _, _, bw, _, bl, yaw = box
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            hw, hl = bw / 2, bl / 2
            local  = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl], [-hw, -hl]])
            rx     = local[:, 0] * cos_y + local[:, 1] * sin_y + bx
            rz     = -local[:, 0] * sin_y + local[:, 1] * cos_y + bz
            ax.fill(rx, rz, color=col, alpha=0.15)
            ax.plot(rx, rz, color=col, lw=1.5, alpha=0.9)
            # Front edge highlight
            ax.plot(rx[:2], rz[:2], color=col, lw=2.5, alpha=1.0)

        # Class label dot
        ax.scatter([bx], [bz], color=col, s=20, zorder=5,
                   edgecolors="white", linewidths=0.5)

        # Distance label
        ax.text(bx + 0.4, bz, f"{det['distance_m']:.0f}m",
                color=col, fontsize=5.5, fontfamily="monospace",
                va="center", alpha=0.85)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=[c / 255 for c in v], label=k)
        for k, v in CLASS_COLORS_PIL.items()
        if any(d["class"] == k for d in detections)
    ]
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right",
                  fontsize=6, framealpha=0.3, facecolor="#0f172a",
                  edgecolor="#334155", labelcolor="white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.tick_params(colors="#475569", labelsize=6)
    ax.set_xlabel("X (m)", color="#64748b", fontsize=7)
    ax.set_ylabel("Z forward (m)", color="#64748b", fontsize=7)
    ax.grid(True, color="#1e293b", lw=0.4, alpha=0.5)
    fig.tight_layout(pad=0.4)

    return _fig_to_base64(fig)
