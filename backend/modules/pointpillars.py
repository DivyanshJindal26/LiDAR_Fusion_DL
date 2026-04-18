"""
PointPillars (OpenPCDet) wrapper — lazy-loaded at import time.

Configure via environment variables:
  OPENPCDET_PATH      path to OpenPCDet repo root
  PP_CHECKPOINT_PATH  path to pointpillar_7728.pth (or similar)
  PP_CONFIG_PATH      path to tools/cfgs/kitti_models/pointpillar.yaml

If any variable is missing or loading fails, run_pointpillars() returns []
and the pipeline falls back to DBSCAN+OBB only.
"""
import os
import sys
import numpy as np

OPENPCDET_PATH = os.environ.get("OPENPCDET_PATH", "")
PP_CKPT        = os.environ.get("PP_CHECKPOINT_PATH", "")
PP_CFG         = os.environ.get("PP_CONFIG_PATH", "")

_pp_model = None
_pp_cfg   = None
_device   = "cpu"

LABEL_MAP = {1: "car", 2: "pedestrian", 3: "cyclist"}


def _load_model():
    global _pp_model, _pp_cfg, _device

    if not OPENPCDET_PATH or not PP_CKPT or not PP_CFG:
        print("[pointpillars] env vars not set — OBB-only fallback mode", flush=True)
        return

    try:
        import torch
        sys.path.insert(0, OPENPCDET_PATH)

        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.datasets import DatasetTemplate
        from pcdet.utils import common_utils

        logger = common_utils.create_logger()
        old_cwd = os.getcwd()
        tools_dir = os.path.join(OPENPCDET_PATH, "tools")
        os.chdir(tools_dir)
        cfg_from_yaml_file(PP_CFG, cfg)
        os.chdir(old_cwd)

        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=DatasetTemplate(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                training=False,
                root_path=".",
                logger=logger,
            ),
        )
        model.load_params_from_file(filename=PP_CKPT, logger=logger, to_cpu=True)

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        if _device == "cuda":
            model.cuda()
        else:
            model.cpu()
        model.eval()

        _pp_model = model
        _pp_cfg   = cfg
        print(f"[pointpillars] loaded on {_device} ✓  classes={cfg.CLASS_NAMES}", flush=True)

    except Exception as exc:
        print(f"[pointpillars] failed to load: {exc} — OBB-only fallback mode", flush=True)


_load_model()


def run_pointpillars(pts_xyz: np.ndarray, score_thresh: float = 0.4) -> list:
    """
    Run PointPillars on ground-removed, forward-facing LiDAR points.

    pts_xyz : (N, 3) numpy float32 — x(fwd), y(left), z(up), already filtered
    Returns : list of {label, score, center(3), dims(3), corners(8,3), heading}
              Empty list if model not loaded (OBB fallback takes over).
    """
    if _pp_model is None:
        return []

    try:
        import torch
        from pcdet.models import load_data_to_gpu

        # Exact notebook code — pad intensity column with zeros
        pts_4 = np.hstack([
            pts_xyz.astype(np.float32),
            np.zeros((len(pts_xyz), 1), dtype=np.float32),
        ])

        input_dict = {"points": pts_4, "frame_id": 0, "calib": None}
        data_dict  = _pp_model.dataset.prepare_data(data_dict=input_dict)
        data_dict  = _pp_model.dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            pred_dicts, _ = _pp_model.forward(data_dict)

        pred   = pred_dicts[0]
        boxes  = pred["pred_boxes"].cpu().numpy()   # (M, 7): x,y,z,l,w,h,yaw
        scores = pred["pred_scores"].cpu().numpy()  # (M,)
        labels = pred["pred_labels"].cpu().numpy()  # (M,) 1-indexed

        from modules.fusion_pp import box_lwh_center_to_corners

        detections = []
        for i in range(len(boxes)):
            if scores[i] < score_thresh:
                continue
            x, y, z, l, w, h, yaw = boxes[i]
            label   = LABEL_MAP.get(int(labels[i]), "unknown")
            center  = np.array([x, y, z], dtype=np.float32)
            dims    = np.array([l, w, h], dtype=np.float32)
            corners = box_lwh_center_to_corners(center, l, w, h, yaw)
            detections.append({
                "label":   label,
                "score":   float(scores[i]),
                "center":  center,
                "dims":    dims,
                "corners": corners,
                "heading": float(yaw),
            })

        return detections

    except Exception as exc:
        print(f"[pointpillars] inference error: {exc}", flush=True)
        return []
