# LiDAR + Camera Fusion for 3D Perception

A real-time 3D object detection system that fuses Velodyne LiDAR point clouds with RGB camera frames. Built for DL Hackathon PS5 on the KITTI dataset. Target: under 50ms end-to-end per frame.

The core contribution is a three-tier hybrid fusion pipeline that tries to get the best out of both sensors at every detection. LiDAR gives accurate geometry but struggles with sparse returns on small objects and at range. The camera gives rich appearance and class information but cannot measure depth. The pipeline is designed so that neither sensor's failures cascade into missed detections.

---

## Calibration

Before any fusion can happen, every LiDAR point needs to land at its correct pixel in the camera image. KITTI provides three matrices for this.

`Tr_velo_to_cam` is a 3x4 rigid-body transform that moves a point from the Velodyne coordinate frame (x forward, y left, z up) into the camera coordinate frame (x right, y down, z forward). `R0_rect` is a 3x3 rotation that accounts for the slight misalignment between the physical cameras in the stereo rig. `P2` is the 3x4 intrinsic projection matrix for camera 2 (left colour camera).

The pipeline collapses these into a single `T_velo_to_img` matrix:

```
T_velo_to_img = P2 @ R0_rect_4x4 @ Tr_velo_to_cam_4x4
```

A LiDAR point `(x, y, z)` in homogeneous form is projected as:

```
[u * d, v * d, d] = T_velo_to_img @ [x, y, z, 1]^T
u = proj[0] / depth
v = proj[1] / depth
```

Points are discarded if `depth <= 0` (behind the camera), or if `(u, v)` falls outside the image bounds. Both the object detection and raw KITTI calib formats are normalised to the same key names (`P2`, `R0_rect`, `Tr_velo_to_cam`) before entering `parse_calib`, so the rest of the pipeline is format-agnostic.

---

## Detection — Camera branch

YOLOv8l runs on the full RGB image at its native resolution. The output is a set of 2D bounding boxes with class labels and confidence scores. Classes outside the set `{car, truck, bus, person, pedestrian, bicycle, cyclist, motorcycle}` are discarded. Boxes below 0.30 confidence are also dropped.

YOLO does not produce any depth information. Its role in the pipeline is to define frustum regions in the image and to confirm or reject LiDAR-derived detections based on appearance.

---

## Detection — LiDAR branch

PointPillars (via OpenPCDet, pretrained on KITTI) runs on the full ground-removed, forward-facing point cloud. Ground removal is a simple height filter: points below `z = -1.5m` in LiDAR frame are dropped. Points behind the vehicle (`x <= 0`) are also removed.

PointPillars divides the bird's-eye view into a grid of vertical columns called pillars. For each pillar it computes a learned feature vector from the points inside it, then runs a 2D convolutional detection head over the feature map. The output is a set of 3D boxes in LiDAR space, each described as `(x, y, z, l, w, h, yaw)` plus a class label and score.

3D box corners are computed using OpenPCDet's native `box_utils.boxes_to_corners_3d`, which handles the rotation correctly. The pipeline falls back to a manual rotation implementation if that utility is unavailable.

PointPillars runs entirely in LiDAR space and has no knowledge of the camera image. Its class accuracy can be lower than YOLO's on ambiguous objects, but its geometry is far more reliable.

---

## Three-tier hybrid fusion

The fusion layer (`fusion_pp.py`) is where LiDAR and camera detections are reconciled. It runs in three sequential tiers, and each detection can only be claimed once.

### Tier 1 — IoM + center-inside fusion

For each PointPillars box with score above 0.40, the pipeline projects its eight 3D corners into the image plane using `T_velo_to_img`. This gives a 2D bounding rectangle in pixel space. It also projects the 3D center point to get `(cx, cy)`.

Each projected PP box is then compared to every unclaimed YOLO box of the same class family:

- car-family: car, truck, bus
- pedestrian-family: person, pedestrian
- cyclist-family: bicycle, motorcycle, cyclist

The match test is:

**Intersection over Minimum (IoM)**: instead of standard IoU, we compute `intersection / min(area_A, area_B)`. This matters because a YOLO box for a partially-occluded object might be much smaller than the projected 3D box. Standard IoU would score this pair low and skip the match. IoM scores it high because the small box is mostly inside the large one.

**Center-inside check**: if the projected 3D center `(cx, cy)` falls inside the YOLO box pixel boundary, the match is accepted regardless of IoM score. This catches cases where the projected corners are noisy or clipped by the image edge but the geometry is otherwise correct.

If either condition is met, the YOLO box is marked as consumed and the PP box gets the YOLO box's 2D bbox as its final `bbox_2d`. These become Tier 1 detections, labelled `HIGH CONF FUSION`.

### Tier 2 — YOLO-gated PP fallback

PointPillars boxes that did not get a Tier 1 match go through a second pass. The confidence threshold drops to 0.25. Instead of IoM, a standard 2D IoU check is used against remaining unclaimed YOLO boxes. If any YOLO box overlaps the projected PP box above IoU threshold 0.25, the PP detection is kept as `pp_gated`. The class-family matching is not enforced here since the geometry check alone is sufficient at this stage.

PP boxes that fail both confidence and the class dimension sanity check (`is_valid_box`) are discarded at this tier. The dimension bounds are class-specific: a car must be between 2.5m and 8.0m long; a pedestrian must be between 1.0m and 2.2m tall.

### Tier 3 — DBSCAN + OBB on remaining YOLO boxes

YOLO detections that were never claimed by PointPillars (either PP missed them or PP was not loaded) still need a 3D estimate. The pipeline uses frustum cropping: all LiDAR points whose projection falls inside the YOLO box pixel boundary are selected. 

DBSCAN clusters those frustum points (eps=0.5m, min_samples=3). The largest cluster is taken. If it has at least 10 points, PCA is run on the XY plane to find the dominant orientation, and an oriented bounding box (OBB) is fitted in the rotated frame. The center, dimensions, and corners come from this OBB.

If the cluster has fewer than 10 points (sparse return, distant object), the pipeline falls back to a class prior: known average dimensions for that class are placed at the cluster centroid. The result is lower quality but prevents the detection from being dropped entirely.

### Merging Tier 2 and Tier 3

Tier 2 (PP-gated) and Tier 3 (OBB/prior) detections are merged. For each PP-gated box, the pipeline looks for an OBB detection within 3.0m center distance. If one exists, the PP box is labelled `HIGH (PP+OBB agree)` — both sensors found something at the same location. If no OBB agrees, the label is `HIGH (PP only, YOLO confirmed)`. OBB detections that had no PP counterpart become `MED (OBB only)`.

### Global 3D NMS

After all three tiers are combined, a global non-maximum suppression pass removes duplicates across tier boundaries. Detections are sorted by priority (Tier 1 first, then Tier 2, then Tier 3) and within each tier by score. Working from highest priority downward, any detection within 2.5m center distance of an already-kept detection is dropped. Priority ordering ensures that a Tier 1 fusion result always wins over a lower-tier result at the same location.

---

## Pipeline flow

```
Raw LiDAR (N, 4)          RGB image (H, W, 3)
       |                          |
  Height filter              YOLOv8l inference
  z > -1.5, x > 0           conf >= 0.30
       |                          |
  PointPillars               2D boxes + classes
  (OpenPCDet)                      |
  score >= 0.40                    |
       |                           |
       +----------+----------------+
                  |
          T_velo_to_img
          project 3D corners
          and center to image
                  |
         Tier 1: IoM >= 0.30
         or center-inside
         (class-aware match)
                  |
         Tier 2: IoU >= 0.25
         on remaining PP + YOLO
         score >= 0.25
                  |
         Tier 3: frustum crop
         DBSCAN -> OBB or prior
         on remaining YOLO
                  |
         Merge T2 + T3
         (3D center distance < 3m)
                  |
         Global 3D NMS
         (center dist < 2.5m,
          priority by tier)
                  |
         Final detections
         with confidence_tier label
```

---

## Detection output

Each final detection carries:

```json
{
  "label":           "car",
  "score":           0.91,
  "confidence_tier": "PERFECT (IoM/Center Fused)",
  "source":          "HIGH CONF FUSION",
  "distance_m":      14.2,
  "center":          [14.1, 0.3, 0.8],
  "dims":            [4.3, 1.9, 1.6],
  "corners":         [[...], ...],
  "heading":         -0.032,
  "bbox_2d":         [120, 200, 340, 380],
  "color_hex":       "#2979ff"
}
```

`distance_m` is Euclidean norm of the 3D center, which in KITTI LiDAR frame is the straight-line distance from the sensor.

Pipeline stats are returned alongside detections:

```json
{
  "yolo_n":        6,
  "pp_raw_n":      9,
  "tier1_fused_n": 4,
  "pp_gated_n":    2,
  "obb_n":         1,
  "final_n":       7
}
```

---

## Bulk / temporal mode

The same pipeline runs per-frame over an entire KITTI ZIP. Two ZIP structures are supported and auto-detected:

**Object detection format** — per-frame calibration files in `calib/000000.txt`:
```
[root/]velodyne/000000.bin
[root/]image_2/000000.png
[root/]calib/000000.txt
```

**KITTI raw format** — single date-level calibration shared across all frames:
```
[date/]calib_cam_to_cam.txt
[date/]calib_velo_to_cam.txt
[date/drive_sync/]velodyne_points/data/0000000000.bin
[date/drive_sync/]image_02/data/0000000000.png
```

In raw format, `calib_cam_to_cam.txt` and `calib_velo_to_cam.txt` are merged: `P_rect_02` becomes `P2`, `R_rect_00` becomes `R0_rect`, and the `R` + `T` rotation/translation fields from the velo-to-cam file become `Tr_velo_to_cam`.

Processing streams progress via Server-Sent Events so the frontend can show frame-by-frame progress. After all frames are processed, three H.264 MP4 videos are assembled: annotated camera output, LiDAR-on-image overlay, and bird's-eye view. ffmpeg is used for encoding (yuv420p, libx264, faststart); OpenCV avc1/mp4v is the fallback if ffmpeg is unavailable.

---

## Scene memory and chatbot

Every processed frame — whether from single upload, scene picker, or bulk ZIP — is stored in a ChromaDB vector collection after inference. The stored document is a plain text summary of that frame: detection count, class breakdown, closest and farthest object distances, per-object label and distance, and total LiDAR point count. ChromaDB embeds this text using its built-in sentence-transformers model.

The collection persists across server restarts at `./chroma_db`. The header shows a live count of how many scenes are in memory; it briefly flashes green labelled "stored" for 1.8 seconds each time a new frame is added.

The chatbot (`POST /chat`) is a full agentic loop that runs server-side via OpenRouter. The API key never reaches the browser. When you send a message, the frontend builds a system prompt that includes the full context of whatever is currently loaded — then sends that prompt plus the conversation history to the LLM. The LLM can call tools (like querying detections, filtering by distance, counting objects by class) and the loop continues until it produces a plain text final answer, which is streamed back. The tool-use intermediate steps are hidden; you only see the final response.

What the system prompt contains depends on mode:

- **Single frame**: all pipeline stats (YOLO count, PP raw, T1 fused, T2 gated, T3 OBB, final), every detection with label, score, confidence tier, distance, 3D center, and fusion source.
- **Bulk time-series**: a global summary across all frames (class counts, avg/min/max distance per class), a compact one-line entry per frame (frame ID, detection count, class histogram, avg distance), and full detail for whichever frame is selected on the timeline scrubber. Capped at 200 frames for token budget.
- **Bulk independent**: only the currently-viewed frame, same detail as single-frame mode.

The model is configurable via `OPENROUTER_MODEL` in `.env`. Default is `x-ai/grok-3-mini-beta`.

---

## PointPillars fallback

If PointPillars is not loaded (env vars not configured, checkpoint missing, or a CUDA error), `run_pointpillars()` returns an empty list. The pipeline then skips Tier 1 and Tier 2 entirely and runs Tier 3 on all YOLO boxes. Results are lower quality (no true 3D box regression, only DBSCAN+OBB estimates) but the pipeline stays functional. This is the mode that runs without a GPU.

---


## Running locally

### Requirements

- Python 3.10+
- Node.js 18+ (for the web frontend)
- ffmpeg on PATH (recommended for video encoding in bulk mode)
- GPU optional — pipeline falls back to CPU/OBB-only mode

### 1. Clone

```bash
git clone https://github.com/DivyanshJindal26/DLHackathon --recurse-submodules
cd DLHackathon
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Install OpenPCDet

```bash
cd OpenPCDet
pip install -r requirements.txt
python setup.py develop
cd ..
```

### 3. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy and fill in the env file:

```bash
cp .env.example .env
```

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=x-ai/grok-3-mini-beta
APP_URL=http://localhost:5173
FRONTEND_URL=http://localhost:5173
KITTI_DATA_DIR=data/kitti

# PointPillars (leave blank to run in OBB-only fallback mode)
OPENPCDET_PATH=/abs/path/to/OpenPCDet
PP_CHECKPOINT_PATH=/abs/path/to/pointpillar_7728.pth
PP_CONFIG_PATH=/abs/path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml
```

Start the server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api/*` to port 8000.

---

## Backend dependencies

| Package | Role |
|---------|------|
| fastapi + uvicorn | HTTP server |
| ultralytics | YOLOv8 inference |
| OpenPCDet | PointPillars 3D detection |
| opencv-python | image I/O, projection drawing, video encoding |
| scikit-learn | DBSCAN clustering, PCA for OBB |
| numpy | all point cloud math |
| matplotlib | BEV visualisation |
| chromadb | scene vector store for chat context |
| openai | OpenRouter API client (chatbot) |
