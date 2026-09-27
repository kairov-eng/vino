"""Local-feature geometric verification (label crop ↔ catalog label).

Architecture: SuperPoint+LightGlue preferred; CPU fallback = ORB/SIFT + BFMatcher
+ RANSAC homography (TECH_CHOICE.md / TZ ФТ-2.5).

Online: extract query keypoints once, match against each embedding candidate's
label crop; score = inliers under a consistent homography.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Wine
from app.media import resolve_label_path, resolve_photo_path

_MAX_SIDE = 1024
_ORB_FEATURES = 2000
_RATIO_TEST = 0.75
_RANSAC_REPROJ = 5.0
_MIN_MATCHES = 8


def _resize_long_side(gray: np.ndarray, max_side: int = _MAX_SIDE) -> np.ndarray:
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray
    scale = max_side / float(m)
    return cv2.resize(
        gray,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _load_gray(path: Path) -> np.ndarray | None:
    if not path or not Path(path).is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _resize_long_side(gray)


def _make_detector():
    """Prefer SIFT (more stable), fall back to ORB."""
    if hasattr(cv2, "SIFT_create"):
        try:
            return "SIFT", cv2.SIFT_create(nfeatures=_ORB_FEATURES)
        except Exception:  # noqa: BLE001
            pass
    return "ORB", cv2.ORB_create(nfeatures=_ORB_FEATURES)


def _extract(gray: np.ndarray, detector) -> tuple[list, np.ndarray | None]:
    kps, desc = detector.detectAndCompute(gray, None)
    if desc is None or len(kps) < 4:
        return list(kps or []), None
    return list(kps), desc


def match_pair(
    query_kps,
    query_desc: np.ndarray,
    ref_gray: np.ndarray,
    detector,
    method: str,
) -> dict[str, Any]:
    """Match query descriptors against one reference image."""
    t0 = time.perf_counter()
    ref_kps, ref_desc = _extract(ref_gray, detector)
    if ref_desc is None or query_desc is None:
        return {
            "ok": False,
            "error": "too_few_keypoints",
            "keypoints_ref": len(ref_kps),
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    if method == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        knn = matcher.knnMatch(query_desc, ref_desc, k=2)
    except cv2.error as exc:
        return {
            "ok": False,
            "error": str(exc),
            "keypoints_ref": len(ref_kps),
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < _RATIO_TEST * n.distance:
            good.append(m)

    matches_n = len(good)
    inliers = 0
    if matches_n >= _MIN_MATCHES:
        src = np.float32([query_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([ref_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _H, mask = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_REPROJ)
        if mask is not None:
            inliers = int(mask.ravel().sum())

    ratio = (inliers / matches_n) if matches_n else 0.0
    # Combined score 0..1: reward both absolute inliers and ratio
    score = min(1.0, (inliers / 80.0) * 0.7 + ratio * 0.3)

    return {
        "ok": True,
        "keypoints_ref": len(ref_kps),
        "matches": matches_n,
        "inliers": inliers,
        "inlier_ratio": round(ratio, 4),
        "score": round(score, 4),
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def _resolve_ref_path(wine: Wine) -> Path | None:
    path = resolve_label_path(wine.photo_name, slug=wine.slug)
    if path is not None:
        return path
    return resolve_photo_path(wine.photo_name, slug=wine.slug)


def verify_candidates_geometry(
    db: Session,
    *,
    query_label_path: Path,
    candidate_ids: list[int],
) -> dict[str, Any]:
    """Compare query label crop to each candidate wine label via local features."""
    t0 = time.perf_counter()
    method, detector = _make_detector()
    query_gray = _load_gray(query_label_path)
    if query_gray is None:
        return {
            "ok": False,
            "error": "cannot_load_query_label",
            "method": method,
            "items": [],
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    query_kps, query_desc = _extract(query_gray, detector)
    if query_desc is None:
        return {
            "ok": False,
            "error": "query_too_few_keypoints",
            "method": method,
            "keypoints_query": len(query_kps),
            "items": [],
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    wines = {
        w.id: w
        for w in db.scalars(select(Wine).where(Wine.id.in_(candidate_ids or []))).all()
    }

    items: list[dict[str, Any]] = []
    for wid in candidate_ids:
        wine = wines.get(int(wid))
        entry: dict[str, Any] = {"id": int(wid), "ok": False}
        if wine is None:
            entry["error"] = "wine_not_found"
            items.append(entry)
            continue
        ref_path = _resolve_ref_path(wine)
        if ref_path is None:
            entry["error"] = "no_label_image"
            items.append(entry)
            continue
        ref_gray = _load_gray(ref_path)
        if ref_gray is None:
            entry["error"] = "cannot_load_ref"
            items.append(entry)
            continue
        match = match_pair(query_kps, query_desc, ref_gray, detector, method)
        entry.update(match)
        entry["ref_path"] = str(ref_path.name)
        items.append(entry)

    items_sorted = sorted(
        items,
        key=lambda x: (-float(x.get("score") or 0), -int(x.get("inliers") or 0), x["id"]),
    )

    return {
        "ok": True,
        "method": method,
        "keypoints_query": len(query_kps),
        "candidate_count": len(candidate_ids),
        "items": items_sorted,
        "by_id": {int(x["id"]): x for x in items},
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def attach_geometry_to_items(
    items: list[dict[str, Any]],
    geometry_by_id: dict[int, dict[str, Any]],
    *,
    resort: bool = True,
) -> list[dict[str, Any]]:
    """Merge geometry fields into candidate items; optionally re-rank by score."""
    out: list[dict[str, Any]] = []
    for it in items:
        wid = int(it["id"])
        g = geometry_by_id.get(wid) or {}
        merged = {
            **it,
            "geometry_ok": bool(g.get("ok")),
            "geometry_inliers": int(g.get("inliers") or 0),
            "geometry_matches": int(g.get("matches") or 0),
            "geometry_inlier_ratio": float(g.get("inlier_ratio") or 0),
            "geometry_score": float(g.get("score") or 0),
        }
        out.append(merged)
    if resort:
        out.sort(
            key=lambda x: (
                -float(x.get("geometry_score") or 0),
                -float(x.get("cosine_similarity") or 0),
                int(x["id"]),
            )
        )
    return out
