"""
ChromaDB scene store for RAG queries.

Each processed frame is stored as a text document (scene summary) in a
persistent ChromaDB collection. ChromaDB's built-in sentence-transformers
embedding is used — no external embedding API needed.

Query via POST /rag-query: semantic similarity search returns top-K matching
frames, then OpenRouter generates a natural-language answer from the context.
"""
import json
import os
import numpy as np

_CHROMA_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
_client = None
_col    = None


def _get_collection():
    global _client, _col
    if _col is not None:
        return _col
    try:
        import chromadb
        _client = chromadb.PersistentClient(path=_CHROMA_PATH)
        _col    = _client.get_or_create_collection(
            "kitti_scenes",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[chroma] collection ready — {_col.count()} docs", flush=True)
    except Exception as exc:
        print(f"[chroma] failed to initialise: {exc}", flush=True)
        _col = None
    return _col


def _build_scene_text(frame_id: str, detections: list, num_points: int) -> str:
    if not detections:
        return f"Frame {frame_id}: no detections. Points: {num_points}."

    parts = []
    for d in detections:
        dist = d.get("distance_m", round(float(np.linalg.norm(d.get("center", [0, 0, 10]))), 1))
        src  = d.get("source", "")
        tier = d.get("confidence_tier", "")
        parts.append(f"{d['label']} at {dist}m [{src}]")

    classes_count: dict = {}
    for d in detections:
        classes_count[d["label"]] = classes_count.get(d["label"], 0) + 1
    cls_summary = ", ".join(f"{v}×{k}" for k, v in classes_count.items())

    dists = [d.get("distance_m", 0.0) for d in detections]
    return (
        f"Frame {frame_id}: {len(detections)} detections ({cls_summary}). "
        f"Closest: {min(dists):.1f}m, Farthest: {max(dists):.1f}m. "
        f"Objects: {', '.join(parts)}. "
        f"LiDAR points: {num_points}."
    )


def store_scene(frame_id: str, detections: list, num_points: int) -> None:
    col = _get_collection()
    if col is None:
        return

    try:
        text   = _build_scene_text(frame_id, detections, num_points)
        dists  = [d.get("distance_m", 0.0) for d in detections] or [0.0]
        labels = list(set(d["label"] for d in detections))

        col.upsert(
            ids=[frame_id],
            documents=[text],
            metadatas=[{
                "frame_id":    frame_id,
                "num_objects": len(detections),
                "classes":     json.dumps(labels),
                "min_dist":    round(min(dists), 2),
                "max_dist":    round(max(dists), 2),
                "num_points":  num_points,
            }],
        )
        print(f"[chroma] stored {frame_id} ({len(detections)} dets)", flush=True)
    except Exception as exc:
        print(f"[chroma] store error: {exc}", flush=True)


def query_scenes(query_text: str, n_results: int = 5) -> list[dict]:
    col = _get_collection()
    if col is None or col.count() == 0:
        return []

    try:
        n = min(n_results, col.count())
        results = col.query(query_texts=[query_text], n_results=n)
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        return [{"text": d, **m} for d, m in zip(docs, metas)]
    except Exception as exc:
        print(f"[chroma] query error: {exc}", flush=True)
        return []
