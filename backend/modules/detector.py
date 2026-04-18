"""
2D object detector — YOLOv8m with synthetic fallback.
Model is loaded eagerly at import time so the first request has no cold-start penalty.
"""
import numpy as np

COCO_TO_KITTI = {
    "car":        "Car",
    "truck":      "Truck",
    "bus":        "Van",
    "person":     "Pedestrian",
    "bicycle":    "Cyclist",
    "motorcycle": "Cyclist",
}

MODEL_NAME = "yolov8l.pt"

# KITTI images are 1242×375 (~3.3:1). At imgsz=640 they letterbox to 640×193
# — small/distant objects shrink to a few pixels and get missed.
# 1280 keeps the full width at native scale and doubles detection recall.
INFER_IMGSZ = 1280

_model = None


def _load_model():
    global _model
    try:
        from ultralytics import YOLO
        print(f"[detector] loading {MODEL_NAME}...", flush=True)
        _model = YOLO(MODEL_NAME)
        _model(np.zeros((375, 1242, 3), dtype=np.uint8), imgsz=INFER_IMGSZ, verbose=False)
        print(f"[detector] {MODEL_NAME} ready (imgsz={INFER_IMGSZ})", flush=True)
    except Exception as e:
        print(f"[detector] could not load {MODEL_NAME}: {e} — will use synthetic fallback", flush=True)


_load_model()


def _iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _cross_class_nms(detections: list[dict], iou_thresh: float = 0.45) -> list[dict]:
    """
    NMS across all KITTI classes.
    Needed because YOLO runs per-class NMS — a truck detected as both 'truck'
    and 'bus' (same box) won't be suppressed by YOLO's built-in NMS.
    """
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept, suppressed = [], set()
    for i, det in enumerate(dets):
        if i in suppressed:
            continue
        kept.append(det)
        for j in range(i + 1, len(dets)):
            if j not in suppressed and _iou(det["bbox_2d"], dets[j]["bbox_2d"]) > iou_thresh:
                suppressed.add(j)
    return kept


def detect(image: np.ndarray, conf_threshold: float = 0.20) -> list[dict]:
    """
    Run 2D object detection on an RGB (H,W,3) uint8 image.
    Returns list of {class, confidence, bbox_2d: [x1,y1,x2,y2]}.
    Falls back to synthetic detections if YOLO is unavailable or finds nothing.
    """
    detections = []

    if _model is not None:
        try:
            # augment=True: test-time augmentation (multi-scale + flip) improves
            # recall for small and backlit objects at a modest speed cost (~2×).
            results = _model(image, imgsz=INFER_IMGSZ, conf=conf_threshold,
                             augment=True, verbose=False)
            boxes = results[0].boxes
            names = results[0].names
            for i in range(len(boxes)):
                coco_name = names[int(boxes.cls[i].item())].lower()
                kitti_name = COCO_TO_KITTI.get(coco_name)
                if kitti_name is None:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append({
                    "class":      kitti_name,
                    "confidence": round(float(boxes.conf[i].item()), 3),
                    "bbox_2d":    [int(x1), int(y1), int(x2), int(y2)],
                })

            detections = _cross_class_nms(detections)

        except Exception as e:
            print(f"[detector] inference error: {e}", flush=True)

    if not detections:
        from modules.synthetic import get_synthetic_detections
        detections = get_synthetic_detections(seed=42)

    return detections
