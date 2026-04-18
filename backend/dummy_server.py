"""
Dummy backend for the LiDAR Fusion frontend.
Generates realistic-looking fake inference results so the UI can be fully tested
without any real model or KITTI data.
"""
import base64
import io
import json
import os
import random
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="LiDAR Fusion Dummy API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── OpenRouter client (chat endpoint) ─────────────────────────────────────────
_oai = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "LiDAR Fusion Demo",
    },
)
MODEL = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-3-mini-beta")

# ── Dummy scene data ───────────────────────────────────────────────────────────
DUMMY_SCENES = ["scene_0001", "scene_0002", "scene_0003", "scene_0007", "scene_0015"]

# Deterministic per scene so the UI feels consistent
SCENE_SEEDS = {s: i * 42 for i, s in enumerate(DUMMY_SCENES)}

CLASSES = ["Car", "Car", "Car", "Pedestrian", "Pedestrian", "Cyclist", "Van", "Truck"]
CLASS_COLORS_PIL = {
    "Car":        (96,  165, 250),   # blue-400
    "Pedestrian": (52,  211, 153),   # emerald-400
    "Cyclist":    (251, 191, 36),    # amber-400
    "Van":        (167, 139, 250),   # violet-400
    "Truck":      (244, 114, 182),   # pink-400
}

IMG_W, IMG_H = 1242, 375  # KITTI camera resolution


def _color(cls: str) -> tuple[int, int, int]:
    return CLASS_COLORS_PIL.get(cls, (148, 163, 184))


def _generate_detections(seed: int, n: int = 6) -> list[dict]:
    rng = random.Random(seed)
    dets = []
    for _ in range(n):
        cls = rng.choice(CLASSES)
        dist = round(rng.uniform(3, 45), 1)
        conf = round(rng.uniform(0.52, 0.99), 2)

        # 2D bbox — wider for cars/vans, narrow for peds
        w_range = (40, 180) if cls in ("Car", "Van", "Truck") else (20, 55)
        h_range = (40, 110) if cls in ("Car", "Van", "Truck") else (50, 130)
        bw = rng.randint(*w_range)
        bh = rng.randint(*h_range)
        x1 = rng.randint(20, IMG_W - bw - 20)
        y1 = rng.randint(IMG_H // 3, IMG_H - bh - 10)
        x2, y2 = x1 + bw, y1 + bh

        # 3D position (LiDAR frame: X=right, Y=down, Z=forward)
        cx = round(rng.uniform(-8, 8), 2)
        cy = round(rng.uniform(-1.5, 0.5), 2)
        cz = round(dist, 2)

        # 3D box: [cx, cy, cz, w, h, l, yaw]
        bw3 = round(rng.uniform(1.6, 2.2), 2)
        bh3 = round(rng.uniform(1.4, 1.9), 2)
        bl3 = round(rng.uniform(3.5, 5.0), 2) if cls in ("Car", "Van", "Truck") else round(rng.uniform(0.4, 0.8), 2)
        yaw = round(rng.uniform(-0.4, 0.4), 3)

        dets.append({
            "class":      cls,
            "confidence": conf,
            "distance_m": dist,
            "bbox_2d":    [x1, y1, x2, y2],
            "xyz":        [cx, cy, cz],
            "box_3d":     [cx, cy, cz, bw3, bh3, bl3, yaw],
        })

    # Sort by distance ascending
    dets.sort(key=lambda d: d["distance_m"])
    return dets


def _draw_annotated_image(dets: list[dict]) -> str:
    """Create a synthetic road-scene image with bounding boxes and return as base64 PNG."""
    img = Image.new("RGB", (IMG_W, IMG_H), color=(30, 30, 40))
    draw = ImageDraw.Draw(img)

    # ── Road background ───────────────────────────────────────────────────────
    # Sky gradient
    for y in range(IMG_H // 2):
        t = y / (IMG_H // 2)
        r = int(15 + 30 * t)
        g = int(15 + 35 * t)
        b = int(25 + 50 * t)
        draw.line([(0, y), (IMG_W, y)], fill=(r, g, b))

    # Road surface
    road_top = IMG_H // 2
    for y in range(road_top, IMG_H):
        t = (y - road_top) / (IMG_H - road_top)
        shade = int(45 + 20 * t)
        draw.line([(0, y), (IMG_W, y)], fill=(shade, shade, shade - 5))

    # Lane markings (dashed white)
    for x in range(0, IMG_W, 80):
        draw.rectangle([x, IMG_H * 2 // 3, x + 40, IMG_H * 2 // 3 + 4], fill=(220, 220, 180))

    # Horizon line (subtle)
    draw.line([(0, road_top), (IMG_W, road_top)], fill=(60, 65, 80), width=1)

    # ── Bounding boxes ────────────────────────────────────────────────────────
    for det in dets:
        x1, y1, x2, y2 = det["bbox_2d"]
        color = _color(det["class"])
        dist = det["distance_m"]

        # Distance-tinted fill
        alpha_fill = Image.new("RGBA", img.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(alpha_fill)
        fill_draw.rectangle([x1, y1, x2, y2], fill=(*color, 30))
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, alpha_fill)
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)

        # Border
        lw = 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        # Label
        label = f"{det['class']} {dist:.1f}m"
        lx, ly = x1 + 3, max(y1 - 14, 0)
        draw.rectangle([lx - 1, ly, lx + len(label) * 6 + 2, ly + 13], fill=(*color, 220))
        draw.text((lx + 1, ly + 1), label, fill=(10, 10, 10))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _draw_bev_image(dets: list[dict], num_points: int, seed: int) -> str:
    """Create a bird's eye view matplotlib figure and return as base64 PNG."""
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#020617")
    ax.set_facecolor("#020617")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-5, 50)
    ax.set_aspect("equal")

    # ── LiDAR point cloud (sparse representation) ────────────────────────────
    # Ground points
    gx = rng.uniform(-24, 24, num_points // 8)
    gz = rng.uniform(0, 48, num_points // 8)
    intensity = rng.uniform(0.05, 0.3, num_points // 8)
    ax.scatter(gx, gz, c=intensity, cmap="Blues", s=0.3, alpha=0.4, vmin=0, vmax=1)

    # Object cluster points around each detection
    for det in dets:
        cx, _, cz = det["xyz"]
        px = rng.normal(cx, 0.6, 40)
        pz = rng.normal(cz, 0.8, 40)
        col = [c / 255 for c in _color(det["class"])]
        ax.scatter(px, pz, color=col, s=2, alpha=0.7)

    # ── Distance rings ────────────────────────────────────────────────────────
    ring_data = [(10, "#ef4444"), (20, "#f97316"), (30, "#84cc16"), (40, "#22c55e")]
    theta = np.linspace(0, 2 * np.pi, 360)
    for r, c in ring_data:
        ax.plot(r * np.sin(theta), r * np.cos(theta),
                color=c, lw=0.6, ls="--", alpha=0.5)
        ax.text(0.5, r + 0.5, f"{r}m", color=c, fontsize=6, alpha=0.7,
                ha="center", fontfamily="monospace")

    # ── Ego vehicle ──────────────────────────────────────────────────────────
    ax.scatter([0], [0], marker="s", s=60, color="#3b82f6",
               edgecolors="#60a5fa", linewidths=1.5, zorder=5)

    # ── Detection box footprints + markers ───────────────────────────────────
    for det in dets:
        cx, _, cz = det["xyz"]
        col = [c / 255 for c in _color(det["class"])]
        box = det.get("box_3d")
        if box:
            bx, _, bz, bw, _, bl, yaw = box
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            hw, hl = bw / 2, bl / 2
            corners = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl], [-hw, -hl]])
            rx = corners[:, 0] * cos_y - corners[:, 1] * sin_y + bx
            rz = corners[:, 0] * sin_y + corners[:, 1] * cos_y + bz
            ax.plot(rx, rz, color=col, lw=1.2, alpha=0.85)
        ax.scatter([cx], [cz], color=col, s=18, zorder=4,
                   edgecolors="black", linewidths=0.4)

    # ── Styling ───────────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.tick_params(colors="#475569", labelsize=6)
    ax.set_xlabel("X (m)", color="#64748b", fontsize=7)
    ax.set_ylabel("Z forward (m)", color="#64748b", fontsize=7)
    ax.grid(True, color="#1e293b", lw=0.5, alpha=0.6)
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="#020617")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _run_dummy_inference(seed: int) -> dict:
    t0 = time.perf_counter()
    n_dets = random.Random(seed).randint(4, 8)
    num_points = random.Random(seed + 1).randint(80_000, 130_000)
    dets = _generate_detections(seed, n_dets)
    ann_img = _draw_annotated_image(dets)
    bev_img = _draw_bev_image(dets, num_points, seed)
    elapsed = int((time.perf_counter() - t0) * 1000)
    # Spoof realistic inference time (the real pipeline targets <50ms)
    fake_ms = random.Random(seed + 7).randint(28, 47)
    return {
        "annotated_image":  ann_img,
        "bev_image":        bev_img,
        "detections":       dets,
        "inference_time_ms": fake_ms,
        "num_points":        num_points,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/infer")
async def infer(
    bin_file: UploadFile = File(...),
    image_file: UploadFile = File(...),
    calib_file: UploadFile = File(...),
):
    seed = hash(bin_file.filename or "default") % (2**31)
    return _run_dummy_inference(seed)


@app.get("/scenes")
async def list_scenes():
    return DUMMY_SCENES


@app.post("/infer-scene/{scene_id}")
async def infer_scene(scene_id: str):
    seed = SCENE_SEEDS.get(scene_id, 1337)
    return _run_dummy_inference(seed)


class QueryRequest(BaseModel):
    text: str
    max_distance_m: float | None = None


@app.post("/query")
async def query_scene(req: QueryRequest):
    # Return the last dummy scene's detections filtered by the query keywords
    seed = 42
    dets = _generate_detections(seed, 6)
    keywords = req.text.lower().split()
    results = [
        d for d in dets
        if any(k in d["class"].lower() for k in keywords)
        or any(k in ("car", "vehicle", "object") for k in keywords)
    ]
    if req.max_distance_m is not None:
        results = [d for d in results if d["distance_m"] <= req.max_distance_m]
    return {"results": results, "query": req.text}


# ── Chat (real OpenRouter call) ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    scene_context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    done: bool = True


@app.post("/chat")
async def chat(req: ChatRequest):
    if not os.environ.get("OPENROUTER_API_KEY"):
        # Fallback: mock response when no key is configured
        return ChatResponse(
            content="No OPENROUTER_API_KEY set. Add it to backend/.env to enable the AI assistant.",
            done=True,
        )

    system = (req.scene_context or {}).get("system", "")
    tools = (req.scene_context or {}).get("tools")

    messages = req.messages
    if system:
        messages = [{"role": "system", "content": system}] + messages

    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = await _oai.chat.completions.create(**kwargs)
    choice = response.choices[0]
    msg = choice.message

    if msg.tool_calls:
        tool_calls_out = [
            {
                "id":    tc.id,
                "name":  tc.function.name,
                "input": json.loads(tc.function.arguments or "{}"),
            }
            for tc in msg.tool_calls
        ]
        return ChatResponse(content=msg.content or "", tool_calls=tool_calls_out, done=False)

    return ChatResponse(content=msg.content or "", done=True)
