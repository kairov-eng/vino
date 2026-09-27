"""Dominant-color CIEDE2000 (ColorDelta) between query label and catalog crops.

Pipeline: saturated pixels → k-means (k=3) in OpenCV Lab → top cluster →
CIEDE2000 vs query dominant. Lower = closer gamut. Color-agnostic (no
named rose/green rules).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.media import resolve_label_path

_S_MIN = 40
_V_MIN = 30
_V_MAX = 250
_K = 3
_MAX_SAMPLES = 8000


def _load_bgr(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def _sat_mask(hsv: np.ndarray) -> np.ndarray:
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return (s >= _S_MIN) & (v >= _V_MIN) & (v <= _V_MAX)


def opencv_lab_to_cie(lab: np.ndarray) -> np.ndarray:
    """OpenCV Lab (L 0..255, a/b 0..255 centered at 128) → CIE L*a*b*."""
    lab = np.asarray(lab, dtype=np.float64).reshape(3)
    return np.array(
        [lab[0] * (100.0 / 255.0), lab[1] - 128.0, lab[2] - 128.0],
        dtype=np.float64,
    )


def ciede2000(lab1_cv: np.ndarray, lab2_cv: np.ndarray) -> float:
    """CIEDE2000 between two OpenCV-scaled Lab triples."""
    L1, a1, b1 = opencv_lab_to_cie(lab1_cv)
    L2, a2, b2 = opencv_lab_to_cie(lab2_cv)
    kL = kC = kH = 1.0
    C1 = float(np.sqrt(a1 * a1 + b1 * b1))
    C2 = float(np.sqrt(a2 * a2 + b2 * b2))
    Cab = (C1 + C2) / 2.0
    G = (
        0.5 * (1.0 - np.sqrt((Cab**7) / (Cab**7 + 25**7)))
        if Cab > 0
        else 0.0
    )
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = float(np.sqrt(a1p * a1p + b1 * b1))
    C2p = float(np.sqrt(a2p * a2p + b2 * b2))
    h1p = float(np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = float(np.degrees(np.arctan2(b2, a2p)) % 360)
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        dhp = dh
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    Lp = (L1 + L2) / 2.0
    Cp = (C1p + C2p) / 2.0
    if C1p * C2p == 0:
        hp = h1p + h2p
    else:
        hs = h1p + h2p
        if abs(h1p - h2p) > 180:
            hp = (hs + 360) / 2.0 if hs < 360 else (hs - 360) / 2.0
        else:
            hp = hs / 2.0
    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp - 30))
        + 0.24 * np.cos(np.radians(2 * hp))
        + 0.32 * np.cos(np.radians(3 * hp + 6))
        - 0.20 * np.cos(np.radians(4 * hp - 63))
    )
    dRo = 30.0 * np.exp(-(((hp - 275.0) / 25.0) ** 2))
    Rc = 2.0 * np.sqrt((Cp**7) / (Cp**7 + 25**7)) if Cp > 0 else 0.0
    Sl = 1.0 + (0.015 * (Lp - 50.0) ** 2) / np.sqrt(20.0 + (Lp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cp
    Sh = 1.0 + 0.015 * Cp * T
    Rt = -np.sin(np.radians(2.0 * dRo)) * Rc
    return float(
        np.sqrt(
            (dLp / (kL * Sl)) ** 2
            + (dCp / (kC * Sc)) ** 2
            + (dHp / (kH * Sh)) ** 2
            + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
        )
    )


def dominant_lab(bgr: np.ndarray, *, k: int = _K) -> np.ndarray | None:
    """Top k-means cluster center in OpenCV Lab (float64 length-3)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = _sat_mask(hsv)
    pts = lab[m]
    if len(pts) < 50:
        pts = lab.reshape(-1, 3)
    if len(pts) < 3:
        return lab.reshape(-1, 3).mean(axis=0).astype(np.float64)
    if len(pts) > _MAX_SAMPLES:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), _MAX_SAMPLES, replace=False)]
    kk = int(min(k, max(1, len(pts) // 20)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _compact, labels, centers = cv2.kmeans(
        pts, kk, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel()
    best_i = int(np.bincount(labels, minlength=kk).argmax())
    return np.asarray(centers[best_i], dtype=np.float64)


def dominant_lab_from_path(path: Path | None) -> np.ndarray | None:
    bgr = _load_bgr(path)
    if bgr is None:
        return None
    return dominant_lab(bgr)


def score_candidates_color_delta(
    query_path: Path,
    candidates: dict[str, Any],
) -> dict[str, Any]:
    """Compare query dominant Lab to each top-N crop. Sets ``color_delta`` on items."""
    groups: dict[int, list[dict[str, Any]]] = {}
    for branch in ("siglip2", "dinov3"):
        for it in (candidates.get(branch) or {}).get("items") or []:
            if not isinstance(it, dict) or it.get("id") is None:
                continue
            groups.setdefault(int(it["id"]), []).append(it)

    base: dict[str, Any] = {
        "ok": False,
        "metric": "ciede2000_dominant",
        "by_id": {},
        "missing": [],
        "n": 0,
        "winner": None,
        "query_lab_cie": None,
    }
    q_lab = dominant_lab_from_path(query_path)
    if q_lab is None:
        base["error"] = "не удалось прочитать искомую этикетку"
        base["missing"] = sorted(groups)
        return base

    base["query_lab_cie"] = [round(float(x), 2) for x in opencv_lab_to_cie(q_lab)]
    by_id: dict[str, float] = {}
    missing: list[int] = []

    for wid, group in groups.items():
        slug = next((str(it["slug"]) for it in group if it.get("slug")), None)
        path = resolve_label_path(None, slug=slug) if slug else None
        c_lab = dominant_lab_from_path(path)
        if c_lab is None:
            missing.append(wid)
            continue
        dist = round(ciede2000(q_lab, c_lab), 2)
        by_id[str(wid)] = dist
        for it in group:
            it["color_delta"] = dist

    base["ok"] = True
    base["by_id"] = by_id
    base["missing"] = missing
    base["n"] = len(by_id)
    return base
