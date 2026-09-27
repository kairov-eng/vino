"""HSV fingerprint of a label crop and Bhattacharyya distance to catalog crops.

Histogram: H×S, 30×32 bins, ranges H 0..180 and S 0..256, L2-normalized.
Catalog wines store the same hist in ``wines.hsv`` (float32 row-major bytes).
Distance 0 — same gamut, 1 — no overlap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Wine
from app.media import resolve_label_path

_BINS = [30, 32]
_RANGES = [0, 180, 0, 256]
_HSV_SHAPE = (30, 32)
_HSV_NBYTES = 30 * 32 * 4  # float32


def hsv_hist(path: Path | None) -> np.ndarray | None:
    """L2-normalized HSV histogram, or None if the file cannot be read."""
    if path is None or not path.is_file():
        return None
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, _BINS, _RANGES)
    cv2.normalize(hist, hist)
    return np.asarray(hist, dtype=np.float32).reshape(_HSV_SHAPE)


def pack_hsv_hist(hist: np.ndarray) -> bytes:
    """Serialize hist to wines.hsv bytes (float32 C-order 30×32)."""
    return np.asarray(hist, dtype=np.float32).reshape(_HSV_SHAPE).tobytes(order="C")


def unpack_hsv_hist(blob: bytes | bytearray | memoryview | None) -> np.ndarray | None:
    """Deserialize wines.hsv → float32 30×32, or None if empty/invalid."""
    if blob is None:
        return None
    raw = bytes(blob)
    if len(raw) != _HSV_NBYTES:
        return None
    return np.frombuffer(raw, dtype=np.float32).reshape(_HSV_SHAPE).copy()


def bhattacharyya(query_hist: np.ndarray, other_hist: np.ndarray) -> float:
    return round(
        float(cv2.compareHist(query_hist, other_hist, cv2.HISTCMP_BHATTACHARYYA)),
        4,
    )


def score_candidates_hsv(
    query_path: Path,
    candidates: dict[str, Any],
    *,
    db: Session | None = None,
) -> dict[str, Any]:
    """Compare the query label to each top-20 crop. Writes ``hsv`` (distance) onto items.

    Prefers ``item['_hsv']`` (``wines.hsv`` loaded at enrich). If missing —
    compute from catalog crop and optionally persist to ``wines.hsv``.
    """
    groups: dict[int, list[dict[str, Any]]] = {}
    for branch in ("siglip2", "dinov3"):
        for it in (candidates.get(branch) or {}).get("items") or []:
            if not isinstance(it, dict) or it.get("id") is None:
                continue
            groups.setdefault(int(it["id"]), []).append(it)

    base: dict[str, Any] = {
        "ok": False,
        "metric": "bhattacharyya",
        "bins": list(_BINS),
        "by_id": {},
        "missing": [],
        "n": 0,
        "from_db": 0,
        "computed": 0,
        "persisted": 0,
        "winner": None,
    }
    query_hist = hsv_hist(query_path)
    if query_hist is None:
        base["error"] = "не удалось прочитать искомую этикетку"
        base["missing"] = sorted(groups)
        return base

    by_id: dict[str, float] = {}
    missing: list[int] = []
    from_db = 0
    computed = 0
    to_persist: list[tuple[int, bytes]] = []

    for wid, group in groups.items():
        blob = next((it.get("_hsv") for it in group if it.get("_hsv")), None)
        hist = unpack_hsv_hist(blob)
        source = "db"
        if hist is None:
            slug = next((str(it["slug"]) for it in group if it.get("slug")), None)
            path = resolve_label_path(None, slug=slug) if slug else None
            hist = hsv_hist(path)
            source = "computed"
            if hist is not None:
                packed = pack_hsv_hist(hist)
                to_persist.append((wid, packed))
                for it in group:
                    it["_hsv"] = packed
        if hist is None:
            missing.append(wid)
            continue
        if source == "db":
            from_db += 1
        else:
            computed += 1
        dist = bhattacharyya(query_hist, hist)
        by_id[str(wid)] = dist
        for it in group:
            it["hsv"] = dist

    persisted = 0
    if db is not None and to_persist:
        ids = [wid for wid, _ in to_persist]
        wines = {
            int(w.id): w
            for w in db.scalars(select(Wine).where(Wine.id.in_(ids))).all()
        }
        for wid, packed in to_persist:
            wine = wines.get(wid)
            if wine is None:
                continue
            # Keep a valid stored hist; overwrite only if empty/corrupt.
            if unpack_hsv_hist(wine.hsv) is not None:
                continue
            wine.hsv = packed
            db.add(wine)
            persisted += 1
        if persisted:
            try:
                db.commit()
            except Exception:  # noqa: BLE001
                db.rollback()
                persisted = 0

    base["ok"] = True
    base["by_id"] = by_id
    base["missing"] = missing
    base["n"] = len(by_id)
    base["from_db"] = from_db
    base["computed"] = computed
    base["persisted"] = persisted
    return base
