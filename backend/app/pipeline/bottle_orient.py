"""Выравнивание кадра по оси бутылки (после YOLO, до OCR/embed).

Угол: Canny + HoughLinesP внутри YOLO-box (линии около вертикали),
медиана по длине. Fallback — PCA по пикселям рёбер.
Поворот кадра с расширением холста; 180° — если «толстый» конец сверху.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_MIN_DEG = 8.0
DEFAULT_PAD = 0.06
_MAX_TILT_FROM_VERTICAL = 40.0


def _clamp_xyxy(
    xyxy: list[float] | tuple[float, ...],
    w: int,
    h: int,
    *,
    pad: float = DEFAULT_PAD,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1i = max(0, int(x1 - pad * bw))
    y1i = max(0, int(y1 - pad * bh))
    x2i = min(w, int(x2 + pad * bw))
    y2i = min(h, int(y2 + pad * bh))
    if x2i <= x1i + 2:
        x2i = min(w, x1i + 3)
    if y2i <= y1i + 2:
        y2i = min(h, y1i + 3)
    return x1i, y1i, x2i, y2i


def normalize_tilt_deg(ang_from_x: float) -> float:
    """Угол линии от +X → отклонение от вертикали (−90…90]."""
    a = float(ang_from_x) % 180.0
    tilt = a - 90.0
    if tilt <= -90.0:
        tilt += 180.0
    if tilt > 90.0:
        tilt -= 180.0
    return float(tilt)


def bottle_edges_roi(
    bgr: np.ndarray,
    xyxy: list[float] | tuple[float, ...],
    *,
    pad: float = DEFAULT_PAD,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    roi_box = _clamp_xyxy(xyxy, w, h, pad=pad)
    x1, y1, x2, y2 = roi_box
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((max(1, y2 - y1), max(1, x2 - x1)), dtype=np.uint8), roi_box
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    return edges, roi_box


def bottle_axis_tilt_deg(
    bgr: np.ndarray,
    xyxy: list[float] | tuple[float, ...],
    *,
    pad: float = DEFAULT_PAD,
) -> tuple[float | None, dict[str, Any]]:
    """Оценка наклона оси бутылки (°) относительно вертикали."""
    edges, roi_box = bottle_edges_roi(bgr, xyxy, pad=pad)
    rh, rw = edges.shape[:2]
    info: dict[str, Any] = {
        "roi": list(roi_box),
        "method": None,
        "lines_n": 0,
        "edge_px": int(np.count_nonzero(edges)),
    }
    min_len = max(40, min(rh, rw) // 5)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=30,
        minLineLength=min_len,
        maxLineGap=25,
    )
    angles: list[float] = []
    weights: list[float] = []
    if lines is not None:
        arr = np.asarray(lines).reshape(-1, 4)
        for x1l, y1l, x2l, y2l in arr:
            dx = float(x2l) - float(x1l)
            dy = float(y2l) - float(y1l)
            leng = float(np.hypot(dx, dy))
            if leng < 20:
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))
            tilt = normalize_tilt_deg(ang)
            if abs(tilt) > _MAX_TILT_FROM_VERTICAL:
                continue
            mx = 0.5 * (float(x1l) + float(x2l)) / max(rw, 1)
            # рёбра бутылки по бокам важнее декора этикетки в центре
            side = 1.0
            if mx <= 0.22 or mx >= 0.78:
                side = 2.2
            elif 0.35 <= mx <= 0.65:
                side = 0.55  # центр: полоски на label (Nik Weis и т.п.)
            angles.append(tilt)
            weights.append(leng * side)

    if len(angles) >= 2:
        # пик гистограммы по суммарной длине — устойчивее медианы при бимодальности
        a = np.asarray(angles, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        bin_w = 2.0
        bins = np.arange(
            -_MAX_TILT_FROM_VERTICAL, _MAX_TILT_FROM_VERTICAL + bin_w, bin_w
        )
        idx = np.digitize(a, bins) - 1
        scores = np.zeros(len(bins) - 1, dtype=np.float64)
        for i, w in zip(idx, ww):
            if 0 <= int(i) < len(scores):
                scores[int(i)] += float(w)
        peak = int(np.argmax(scores))
        # среднее углов в пиковом бине (взвешенное)
        lo, hi = float(bins[peak]), float(bins[peak + 1])
        mask = (a >= lo) & (a < hi)
        if not np.any(mask):
            mask = (a >= lo - bin_w) & (a < hi + bin_w)
        if np.any(mask):
            tilt_est = float(np.average(a[mask], weights=ww[mask]))
        else:
            tilt_est = float(0.5 * (lo + hi))
        info["method"] = "hough_peak"
        info["lines_n"] = len(angles)
        info["peak_bin"] = [round(lo, 1), round(hi, 1)]
        info["peak_score"] = round(float(scores[peak]), 1)
        return tilt_est, info

    ys, xs = np.where(edges > 0)
    if len(xs) < 80:
        info["method"] = "none"
        return None, info
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    _mean, ev = cv2.PCACompute(pts, mean=None)
    v = ev[0]
    ang = float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))
    info["method"] = "pca_edges"
    return float(normalize_tilt_deg(ang)), info


def _foreground_mask(bgr: np.ndarray, xyxy: list[float] | tuple[float, ...]) -> np.ndarray:
    """Грубая маска для проверки горлышка (верх/низ) после поворота."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _clamp_xyxy(xyxy, w, h, pad=0.02)
    roi = bgr[y1:y2, x1:x2]
    full = np.zeros((h, w), dtype=np.uint8)
    if roi.size == 0:
        return full
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return full
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    full[y1:y2, x1:x2] = np.where(labels == best, 255, 0).astype(np.uint8)
    return full


def _needs_flip_180(mask: np.ndarray) -> bool:
    ys, xs = np.where(mask > 0)
    if len(ys) < 50:
        return False
    y0, y1 = int(ys.min()), int(ys.max())
    hh = y1 - y0 + 1
    if hh < 30:
        return False
    top = mask[y0 : y0 + hh // 3]
    bot = mask[y1 - hh // 3 : y1 + 1]
    top_w = float(np.count_nonzero(top))
    bot_w = float(np.count_nonzero(bot))
    if bot_w <= 1:
        return False
    return top_w > bot_w * 1.15


def rotation_matrix_expand(
    w: int, h: int, angle_deg: float
) -> tuple[np.ndarray, int, int]:
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    cos_a = abs(float(m[0, 0]))
    sin_a = abs(float(m[0, 1]))
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    m[0, 2] += (nw - w) / 2.0
    m[1, 2] += (nh - h) / 2.0
    return m, nw, nh


def warp_bgr(bgr: np.ndarray, m: np.ndarray, nw: int, nh: int) -> np.ndarray:
    return cv2.warpAffine(
        bgr,
        m,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def transform_xyxy(xyxy: list[float], m: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    ones = np.ones((4, 1), dtype=np.float64)
    out = np.hstack([pts, ones]) @ np.asarray(m, dtype=np.float64).T
    xs, ys = out[:, 0], out[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def transform_box_dict(box: dict[str, Any], m: np.ndarray) -> dict[str, Any]:
    out = dict(box)
    if box.get("xyxy"):
        xyxy = transform_xyxy(list(box["xyxy"]), m)
        out["xyxy"] = xyxy
        out["area"] = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
    return out


def upright_bgr_by_bottle(
    bgr: np.ndarray,
    bottle_xyxy: list[float] | tuple[float, ...],
    *,
    min_deg: float = DEFAULT_MIN_DEG,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    h0, w0 = bgr.shape[:2]
    meta: dict[str, Any] = {
        "ok": True,
        "applied": False,
        "tilt_deg": None,
        "rotate_deg": 0.0,
        "flip_180": False,
        "min_deg": float(min_deg),
        "reason": "ok",
        "image": bgr,
        "matrix": None,
        "size_in": [w0, h0],
        "size_out": [w0, h0],
        "method": None,
        "lines_n": 0,
        "ms": 0.0,
    }
    try:
        tilt, info = bottle_axis_tilt_deg(bgr, bottle_xyxy)
        meta["method"] = info.get("method")
        meta["lines_n"] = info.get("lines_n")
        meta["roi"] = info.get("roi")
        meta["edge_px"] = info.get("edge_px")
        if tilt is None:
            meta["reason"] = "no_axis"
            meta["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return meta
        meta["tilt_deg"] = round(float(tilt), 2)
        if abs(float(tilt)) < float(min_deg):
            meta["reason"] = "below_threshold"
            meta["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return meta

        # tilt<0 = верх вправо; OpenCV+ = CCW → крутим на тот же знак, что tilt.
        # На широком ROI (бутылка ≈ весь кадр) Hough завышает угол — выбираем
        # масштаб 0.5…1.0 по минимальному |остатку|.
        scales = (1.0, 0.8, 0.65, 0.5)
        best: tuple[float, np.ndarray, np.ndarray, int, int, float] | None = None
        for sc in scales:
            ang = float(tilt) * float(sc)
            m_try, nw_try, nh_try = rotation_matrix_expand(w0, h0, ang)
            out_try = warp_bgr(bgr, m_try, nw_try, nh_try)
            xy_try = transform_xyxy([float(v) for v in bottle_xyxy], m_try)
            t_res, _info_r = bottle_axis_tilt_deg(out_try, xy_try)
            score = 999.0 if t_res is None else abs(float(t_res))
            if best is None or score < best[0]:
                best = (score, out_try, m_try, nw_try, nh_try, ang)
            if score <= 3.0:
                break
        assert best is not None
        _score, out, m, nw, nh, rotate_deg = best
        meta["residual_deg"] = None if _score >= 900 else round(float(_score), 2)
        meta["scale"] = round(float(rotate_deg) / float(tilt), 3) if tilt else 1.0

        # 180° только для почти лежачих бутылок (|tilt|≫): на полке с наклоном
        # камеры маска Otsu часто врёт «толстый верх».
        if abs(float(tilt)) >= 55.0:
            mask0 = _foreground_mask(bgr, bottle_xyxy)
            mask_r = cv2.warpAffine(
                mask0,
                m,
                (nw, nh),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if _needs_flip_180(mask_r):
                m2, nw2, nh2 = rotation_matrix_expand(nw, nh, 180.0)
                out = warp_bgr(out, m2, nw2, nh2)
                m_3 = np.vstack([m, [0.0, 0.0, 1.0]])
                m2_3 = np.vstack([m2, [0.0, 0.0, 1.0]])
                m = (m2_3 @ m_3)[:2]
                nw, nh = nw2, nh2
                rotate_deg += 180.0
                meta["flip_180"] = True

        meta["applied"] = True
        meta["rotate_deg"] = round(rotate_deg, 2)
        meta["image"] = out
        meta["matrix"] = m
        meta["size_out"] = [nw, nh]
        meta["reason"] = "rotated"
    except Exception as exc:  # noqa: BLE001
        meta["ok"] = False
        meta["reason"] = f"error:{exc}"
    meta["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return meta


def save_bgr_jpeg(path: Path, bgr: np.ndarray, *, quality: int = 85) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def upright_work_image(
    src_path: Path,
    bottle_xyxy: list[float] | tuple[float, ...],
    *,
    out_path: Path | None = None,
    min_deg: float = DEFAULT_MIN_DEG,
    jpeg_quality: int = 85,
) -> dict[str, Any]:
    """Выровнять копию кадра.

    Исходный ``src_path`` не меняется. Если ``applied`` — пишем в ``out_path``
    (обязателен при applied; если None — ошибка, чтобы не затереть оригинал).
    """
    bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        return {
            "ok": False,
            "applied": False,
            "reason": "read_failed",
            "ms": 0.0,
            "matrix": None,
        }
    result = upright_bgr_by_bottle(bgr, bottle_xyxy, min_deg=min_deg)
    if result.get("applied") and result.get("image") is not None:
        if out_path is None:
            result["ok"] = False
            result["applied"] = False
            result["reason"] = "out_path_required"
            result["image"] = None
            result["matrix"] = None
        else:
            save_bgr_jpeg(out_path, result["image"], quality=jpeg_quality)
            result["orient_path"] = str(out_path)
            result["orient_filename"] = out_path.name
    out = {k: v for k, v in result.items() if k != "image"}
    if out.get("matrix") is not None:
        out["matrix"] = [
            [round(float(x), 5) for x in row]
            for row in np.asarray(out["matrix"])
        ]
    return out
