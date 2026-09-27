"""YOLO label detection for wine search pipeline."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ImageDraw, ImageFont

from app.db.config import (
    YOLO_BOTTLE_LABEL_MODEL_PATH,
    YOLO_CONF,
    YOLO_DEVICE,
    YOLO_HALF,
    YOLO_IMGSZ,
    YOLO_MODEL_PATH,
)
from app.pipeline.image_io import open_image_rgb

# Conf gap: if largest is ≥ this below next-by-size → pick next
# (0.9 ≈ почти никогда: conf ∈ [0,1], gap 0.9 почти недостижим)
_LABEL_CONF_SIZE_GAP = 0.9
# Если «самая большая» этикетка слабая — достаточно меньшего gap по conf
_LABEL_CONF_SIZE_GAP_WEAK_LARGEST = 0.15
_LABEL_WEAK_LARGEST_CONF = 0.55
# Не брать «следующую по размеру», если она ≤ половины площади самой большой
_LABEL_MIN_AREA_RATIO = 0.5
# Слабая этикетка на всю высоту бутылки → обычно FP (фон / силуэт), не корпус
_LABEL_FULL_BOTTLE_H_FRAC = 0.85
_LABEL_FULL_BOTTLE_MAX_CONF = 0.55
# Главная бутылка: если на другой max conf этикетки корпуса выше на gap —
# переключаемся (scan 2729: площадь слева vs real label conf 0.96 справа)
_PRIMARY_BOTTLE_LABEL_CONF_GAP = 0.20
_PRIMARY_BOTTLE_LABEL_MIN_CONF = 0.55
_BOX_LABEL_FONT_SIZE = 20
# label∩bottle / area(label) ≥ порог → этикетка «принадлежит» бутылке (merge)
_BOTTLE_LABEL_OVERLAP_RATIO = 0.85
# Сливать все этикетки одной бутылки, если их ≥2 (без верхнего лимита)
_BOTTLE_LABEL_MERGE_MIN = 2
# >10% площади вне бутылки → это не этикетка (игнор для primary)
_LABEL_MAX_OUTSIDE_FRAC = 0.10
_LABEL_MIN_IN_BOTTLE_RATIO = 1.0 - _LABEL_MAX_OUTSIDE_FRAC  # 0.90
# Центр этикетки по высоте бутылки: «корпус» vs «горлышко»
_LABEL_BODY_Y_MIN = 0.20
_LABEL_BODY_Y_MAX = 0.90
_LABEL_NECK_Y_MAX = 0.20
# Fallback-кроп бутылки: отрезать верх (ценники на шее)
_BOTTLE_FALLBACK_TOP_TRIM = 0.18
_BOTTLE_FALLBACK_TOP_TRIM_MAX = 0.35
# Сирота: ≤50% площади в любой бутылке + conf≤0.5 → drop, если есть др. этикетки в бутылках
_ORPHAN_OUTSIDE_RATIO = 0.5  # max in-bottle ratio below/equal → «большей частью вне»
_ORPHAN_MAX_CONF = 0.5
_ORPHAN_PEER_IN_BOTTLE_RATIO = 0.5  # peer «в контуре» бутылки

YOLO_VARIANTS = ("label", "bottle_label")


def _box_label_font(size: int = _BOX_LABEL_FONT_SIZE) -> ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ):
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


_log = logging.getLogger("yolo")

_yolo_models: dict[str, Any] = {}
_yolo_resolved_device: str | None = None
_yolo_use_half: bool = False
_warmup_started = False
_warmup_lock = threading.Lock()


def normalize_yolo_variant(value: Any) -> str:
    raw = str(value or "label").strip().lower().replace("-", "_")
    if raw in {"bottle_label", "bottle+label", "bottlelabel", "bl"}:
        return "bottle_label"
    return "label"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _resolve_device_and_half() -> tuple[str, bool]:
    """Prefer CUDA device from env; fall back to CPU (half only on CUDA)."""
    raw = (YOLO_DEVICE or "0").strip() or "0"
    want_cpu = raw.lower() in {"cpu", "none", "-1"}
    if want_cpu or not _cuda_available():
        if not want_cpu and raw not in {"cpu", "none", "-1"}:
            _log.warning(
                "YOLO: CUDA недоступна — inference на CPU (YOLO_DEVICE=%s)",
                raw,
            )
        return "cpu", False
    # "0" / "cuda" / "cuda:0"
    if raw.lower() in {"cuda", "gpu"}:
        device = "0"
    else:
        device = raw
    half = bool(YOLO_HALF)
    return device, half


def _predict_kwargs() -> dict[str, Any]:
    global _yolo_resolved_device, _yolo_use_half
    if _yolo_resolved_device is None:
        _yolo_resolved_device, _yolo_use_half = _resolve_device_and_half()
    kw: dict[str, Any] = {
        "conf": YOLO_CONF,
        "imgsz": YOLO_IMGSZ,
        "device": _yolo_resolved_device,
        "verbose": False,
    }
    if _yolo_use_half:
        kw["half"] = True
    return kw


def _model_path_for_variant(variant: str) -> Path:
    v = normalize_yolo_variant(variant)
    if v == "bottle_label":
        path = (YOLO_BOTTLE_LABEL_MODEL_PATH or "").strip()
        if not path:
            raise FileNotFoundError(
                "YOLO_BOTTLE_LABEL_MODEL_PATH не задан. "
                "Укажите путь к весам bottle+label."
            )
        return Path(path)
    path = (YOLO_MODEL_PATH or "").strip()
    if not path:
        raise FileNotFoundError(
            "YOLO_MODEL_PATH не задан в настройках (.env). "
            "Укажите путь к обученным весам детекции этикетки."
        )
    return Path(path)


def _load_model(variant: str = "label"):
    p = _model_path_for_variant(variant)
    if not p.is_file():
        raise FileNotFoundError(f"Файл модели YOLO не найден: {p}")
    key = str(p.resolve())
    if key not in _yolo_models:
        from ultralytics import YOLO

        _yolo_models[key] = YOLO(str(p))
        _log.info(
            "YOLO loaded variant=%s file=%s device=%s half=%s",
            normalize_yolo_variant(variant),
            p.name,
            _predict_kwargs().get("device"),
            _predict_kwargs().get("half", False),
        )
    return _yolo_models[key]


def _boxes_from_result(model: Any, r0: Any) -> list[dict[str, Any]]:
    names = getattr(r0, "names", None) or getattr(model, "names", {}) or {}
    boxes_out: list[dict[str, Any]] = []
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    for box in r0.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        cls_name = str(names.get(cls_id, cls_id))
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        boxes_out.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": conf,
                "area": area,
            }
        )
    return boxes_out


def _predict_all_boxes(
    image_path: Path, *, variant: str = "label"
) -> tuple[list[dict[str, Any]], dict[Any, Any]]:
    model = _load_model(variant)
    results = model.predict(source=str(image_path), **_predict_kwargs())
    if not results:
        return [], {}
    r0 = results[0]
    names = getattr(r0, "names", None) or getattr(model, "names", {}) or {}
    return _boxes_from_result(model, r0), names


def _is_label_cls(name: str) -> bool:
    n = str(name).lower()
    return "label" in n or n in {"этикетка", "labels"}


def _is_bottle_cls(name: str) -> bool:
    n = str(name).lower()
    return "bottle" in n or n in {"бутылка", "bottles"}


def _xyxy_area(xyxy: list[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _label_in_bottle_ratio(label: dict[str, Any], bottle: dict[str, Any]) -> float:
    la = float(label.get("area") or _xyxy_area(label["xyxy"]))
    if la <= 0:
        return 0.0
    return _intersection_area(label["xyxy"], bottle["xyxy"]) / la


def _best_bottle_ratio(
    label: dict[str, Any], bottles: list[dict[str, Any]]
) -> float:
    if not bottles:
        return 0.0
    return max(_label_in_bottle_ratio(label, b) for b in bottles)


def filter_orphan_low_conf_labels(
    bottles: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    outside_ratio: float = _ORPHAN_OUTSIDE_RATIO,
    max_conf: float = _ORPHAN_MAX_CONF,
    peer_in_ratio: float = _ORPHAN_PEER_IN_BOTTLE_RATIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop low-conf labels that lie mostly outside every bottle.

    Ignore label L when all hold:
    - conf(L) ≤ max_conf
    - max over bottles of area(L∩bottle)/area(L) ≤ outside_ratio
      (т.е. ≥50% площади вне любой бутылки)
    - есть хотя бы одна другая этикетка с overlap ≥ peer_in_ratio
      хотя бы с одной бутылкой

    Returns ``(kept, ignored)``.
    """
    if not labels or not bottles:
        return list(labels), []

    has_peer_in_bottle = any(
        _best_bottle_ratio(lab, bottles) >= float(peer_in_ratio) for lab in labels
    )
    if not has_peer_in_bottle:
        return list(labels), []

    kept: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for lab in labels:
        conf = float(lab.get("conf") or 0.0)
        best = _best_bottle_ratio(lab, bottles)
        mostly_outside = best <= float(outside_ratio)
        low_conf = conf <= float(max_conf)
        others_in = any(
            p is not lab
            and _best_bottle_ratio(p, bottles) >= float(peer_in_ratio)
            for p in labels
        )
        if mostly_outside and low_conf and others_in:
            drop = dict(lab)
            drop["ignore_reason"] = "orphan_low_conf_outside_bottles"
            drop["best_bottle_ratio"] = round(best, 4)
            ignored.append(drop)
            continue
        kept.append(lab)
    return kept, ignored


def _merge_label_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Unify labels on one bottle: top from topmost, bottom from bottommost."""
    top = min(group, key=lambda b: float(b["xyxy"][1]))
    bot = max(group, key=lambda b: float(b["xyxy"][3]))
    tx1, ty1, tx2, _ty2 = [float(v) for v in top["xyxy"]]
    bx1, _by1, bx2, by2 = [float(v) for v in bot["xyxy"]]
    # Ширину расширяем на все этикетки группы (средняя окажется внутри)
    x1 = min(tx1, bx1, *(float(g["xyxy"][0]) for g in group))
    x2 = max(tx2, bx2, *(float(g["xyxy"][2]) for g in group))
    y1 = ty1
    y2 = by2
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    conf = max(float(g.get("conf") or 0.0) for g in group)
    return {
        "xyxy": [x1, y1, x2, y2],
        "cls_id": int(top.get("cls_id") if top.get("cls_id") is not None else -1),
        "cls_name": "label_merged",
        "conf": conf,
        "area": area,
        "merged": True,
        "merged_count": len(group),
        "merged_from": [
            {
                "xyxy": [round(float(v), 2) for v in g["xyxy"]],
                "conf": round(float(g.get("conf") or 0.0), 4),
            }
            for g in group
        ],
    }


def merge_labels_by_bottles(
    bottles: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    overlap_ratio: float = _BOTTLE_LABEL_OVERLAP_RATIO,
) -> list[dict[str, Any]]:
    """Merge all labels that each overlap the same bottle ≥ overlap_ratio.

    Overlap = area(label ∩ bottle) / area(label).
    If ≥2 labels belong to a bottle, they become one box (top→topmost,
    bottom→bottommost). Single labels stay as-is.
    """
    if not labels:
        return []
    if not bottles:
        return sorted(labels, key=lambda b: float(b.get("area") or 0.0), reverse=True)

    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for bottle in bottles:
        idxs = [
            i
            for i, lab in enumerate(labels)
            if i not in used
            and _label_in_bottle_ratio(lab, bottle) >= float(overlap_ratio)
        ]
        if len(idxs) >= _BOTTLE_LABEL_MERGE_MIN:
            group = [labels[i] for i in idxs]
            out.append(_merge_label_group(group))
            used.update(idxs)

    for i, lab in enumerate(labels):
        if i not in used:
            out.append(lab)
    return sorted(out, key=lambda b: float(b.get("area") or 0.0), reverse=True)


def detect_bottle_and_labels(image_path: Path) -> dict[str, Any]:
    """Bottle+label model: raw bottles/labels + merged label boxes for crop."""
    boxes_out, names = _predict_all_boxes(image_path, variant="bottle_label")
    bottles = [b for b in boxes_out if _is_bottle_cls(str(b["cls_name"]))]
    labels = [b for b in boxes_out if _is_label_cls(str(b["cls_name"]))]

    # Fallback: unknown names — treat non-bottle as labels if class ids known
    if not labels and not bottles and boxes_out:
        name_vals = {str(v).lower() for v in names.values()} if names else set()
        if not names or len(name_vals) <= 1:
            labels = list(boxes_out)

    bottles = sorted(bottles, key=lambda b: float(b.get("area") or 0.0), reverse=True)
    labels = sorted(labels, key=lambda b: float(b.get("area") or 0.0), reverse=True)
    labels_kept, labels_ignored = filter_orphan_low_conf_labels(bottles, labels)
    label_boxes = merge_labels_by_bottles(bottles, labels_kept)
    return {
        "bottles": bottles,
        "labels": labels,
        "labels_kept": labels_kept,
        "labels_ignored": labels_ignored,
        "label_boxes": label_boxes,
        "raw": boxes_out,
    }


def detect_label_boxes(
    image_path: Path, *, variant: str = "label"
) -> list[dict[str, Any]]:
    """Run YOLO and return label boxes as xyxy + score + class.

    Prefers class name/id for «label»; if classes unknown, returns all boxes.
    For ``bottle_label`` variant returns merged label boxes (all labels per bottle).
    """
    v = normalize_yolo_variant(variant)
    if v == "bottle_label":
        return detect_bottle_and_labels(image_path)["label_boxes"]

    boxes_out, names = _predict_all_boxes(image_path, variant="label")
    if not boxes_out:
        return []

    label_boxes = [b for b in boxes_out if _is_label_cls(str(b["cls_name"]))]
    if label_boxes:
        return sorted(label_boxes, key=lambda b: b["area"], reverse=True)

    # Single-class models (class 0 = label) — take all detections
    name_vals = {str(v).lower() for v in names.values()} if names else set()
    if not names or name_vals <= {"label", "0"} or len(name_vals) <= 1:
        return sorted(boxes_out, key=lambda b: b["area"], reverse=True)

    # Multi-class without label hits — return empty (caller handles)
    return []


def warmup_yolo(
    *, imgsz: int | None = None, variant: str | None = None
) -> dict[str, Any]:
    """Load weights + one dummy predict (убирает cold start первого скана)."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {"ok": False}
    variants = (
        [normalize_yolo_variant(variant)]
        if variant is not None
        else list(YOLO_VARIANTS)
    )
    results: dict[str, Any] = {}
    try:
        size = int(imgsz or YOLO_IMGSZ or 640)
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        kw = _predict_kwargs()
        all_ok = True
        for v in variants:
            vt0 = time.perf_counter()
            try:
                model = _load_model(v)
                model.predict(source=blank, **kw)
                results[v] = {
                    "ok": True,
                    "ms": round((time.perf_counter() - vt0) * 1000, 1),
                }
            except Exception as exc:  # noqa: BLE001
                all_ok = False
                results[v] = {"ok": False, "error": str(exc)}
                _log.warning("YOLO warmup failed variant=%s: %s", v, exc)
        out["ok"] = all_ok
        out["device"] = kw.get("device")
        out["half"] = bool(kw.get("half"))
        out["imgsz"] = size
        out["variants"] = results
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        _log.warning("YOLO warmup failed: %s", exc)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if out.get("ok"):
        _log.info(
            "YOLO warmup ok device=%s half=%s ms=%s variants=%s",
            out.get("device"),
            out.get("half"),
            out.get("ms"),
            list((out.get("variants") or {}).keys()),
        )
    return out


def start_yolo_warmup_background() -> None:
    """Non-blocking YOLO load+predict so first scan is not 3–7s cold-start."""
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

    def _job() -> None:
        try:
            warmup_yolo()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_job, name="yolo-warmup", daemon=True).start()


def draw_boxes(
    image_path: Path,
    boxes: list[dict[str, Any]],
    out_path: Path,
    *,
    thickness: int = 5,
    primary: dict[str, Any] | None = None,
    bottles: list[dict[str, Any]] | None = None,
) -> Path:
    """Draw boxes: primary (or largest) yellow, others green. Save to out_path.

    Optional ``bottles`` drawn in cyan with caption ``coef=x.xx`` (top-left).

    Each label box gets top-left caption (2 lines, size vs largest area)::

        conf=x.xx
        size=xx.x%
    """
    img, _ = open_image_rgb(image_path)
    draw = ImageDraw.Draw(img)
    font = _box_label_font(_BOX_LABEL_FONT_SIZE)
    line_gap = 2

    def _place_text(x1: float, y1: float, lines: list[str], color: tuple[int, ...]) -> None:
        heights: list[int] = []
        widths: list[int] = []
        for line in lines:
            tb = draw.textbbox((0, 0), line, font=font)
            widths.append(tb[2] - tb[0])
            heights.append(tb[3] - tb[1])
        tw = max(widths) if widths else 0
        th = sum(heights) + line_gap * max(0, len(lines) - 1)
        tx = int(round(x1)) + 2
        ty = int(round(y1)) + 2
        if tx + tw > img.width:
            tx = max(0, img.width - tw - 1)
        if tx < 0:
            tx = 0
        if ty + th > img.height:
            ty = max(0, img.height - th - 1)
        if ty < 0:
            ty = 0
        yy = ty
        for i, line in enumerate(lines):
            draw.text((tx, yy), line, fill=color, font=font)
            yy += heights[i] + line_gap

    for b in bottles or []:
        x1, y1, x2, y2 = [float(v) for v in b["xyxy"]]
        color = (0, 200, 220)  # cyan
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
        conf = float(b.get("conf") or 0.0)
        _place_text(x1, y1, [f"coef={conf:.2f}"], color)

    if not boxes and not bottles:
        img.save(out_path)
        return out_path
    if not boxes:
        img.save(out_path)
        return out_path

    max_area = max(float(b.get("area") or 0.0) for b in boxes) or 1.0

    ordered = sorted(boxes, key=lambda b: float(b.get("area") or 0.0), reverse=True)
    if primary and primary.get("xyxy") is not None:
        # Ensure primary is drawn first (yellow)
        pxy = [round(float(v), 2) for v in primary["xyxy"]]
        rest = [
            b
            for b in ordered
            if [round(float(v), 2) for v in b["xyxy"]] != pxy
        ]
        ordered = [primary] + rest
    for i, b in enumerate(ordered):
        x1, y1, x2, y2 = [float(v) for v in b["xyxy"]]
        color = (255, 220, 0) if i == 0 else (0, 180, 0)  # yellow / green
        for t in range(thickness):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=color,
            )
        area = float(b.get("area") or 0.0)
        conf = float(b.get("conf") or 0.0)
        size_pct = 100.0 * area / max_area
        _place_text(
            x1,
            y1,
            [f"conf={conf:.2f}", f"size={size_pct:.1f}%"],
            color,
        )
    img.save(out_path)
    return out_path


def select_primary_label(
    boxes: list[dict[str, Any]],
    *,
    similar_ratio: float = 0.12,
    conf_gap: float = _LABEL_CONF_SIZE_GAP,
    min_area_ratio: float = _LABEL_MIN_AREA_RATIO,
) -> dict[str, Any] | None:
    """Pick which label box to crop.

    - Default: largest by area.
    - If exactly 3 boxes and all areas within ``similar_ratio`` of each other
      (max−min)/max ≤ ratio, pick the middle by area.
    - Else if next-by-size has conf ≥ ``conf_gap`` above largest *and*
      its area is > ``min_area_ratio`` of the largest (not ≤ half size),
      pick the next-by-size.
    - Else if largest is weak (conf < 0.55) and next is ≥0.15 higher with
      area > 25% of largest — pick next (scan-style FP «whole bottle» boxes).
    """
    if not boxes:
        return None
    ordered = sorted(boxes, key=lambda b: float(b.get("area") or 0.0), reverse=True)
    if len(ordered) == 3:
        areas = [float(b.get("area") or 0.0) for b in ordered]
        amin, amax = min(areas), max(areas)
        if amin > 0 and (amax - amin) / amax <= similar_ratio:
            # sorted desc → index 1 is the middle size
            pick = dict(ordered[1])
            pick["select_reason"] = "middle_of_three_similar"
            return pick
    if len(ordered) >= 2:
        a0 = float(ordered[0].get("area") or 0.0)
        a1 = float(ordered[1].get("area") or 0.0)
        c0 = float(ordered[0].get("conf") or 0.0)
        c1 = float(ordered[1].get("conf") or 0.0)
        # Вдвое меньше (или ещё меньше) → всегда берём большую
        if a0 > 0 and a1 <= float(min_area_ratio) * a0:
            pick = dict(ordered[0])
            pick["select_reason"] = "largest_vs_half_size"
            pick["select_area_ratio"] = round(a1 / a0, 4) if a0 else None
            return pick
        if c1 - c0 >= float(conf_gap):
            pick = dict(ordered[1])
            pick["select_reason"] = "next_by_size_higher_conf"
            pick["select_conf_gap"] = round(c1 - c0, 4)
            return pick
        # Крупный low-conf бокс (часто весь силуэт) vs меньший high-conf label
        if (
            c0 < float(_LABEL_WEAK_LARGEST_CONF)
            and (c1 - c0) >= float(_LABEL_CONF_SIZE_GAP_WEAK_LARGEST)
            and a0 > 0
            and a1 > 0.25 * a0
        ):
            pick = dict(ordered[1])
            pick["select_reason"] = "next_by_size_weak_largest"
            pick["select_conf_gap"] = round(c1 - c0, 4)
            return pick
    pick = dict(ordered[0])
    pick["select_reason"] = "largest"
    return pick


def select_primary_bottle(
    bottles: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Главная бутылка: больше площадь, при равенстве — выше conf."""
    if not bottles:
        return None
    return max(
        bottles,
        key=lambda b: (
            float(b.get("area") or 0.0),
            float(b.get("conf") or 0.0),
        ),
    )


def _label_is_full_bottle_false_positive(
    label: dict[str, Any],
    bottle: dict[str, Any],
) -> bool:
    """Low-conf box spanning almost the full bottle height — not a real label."""
    conf = float(label.get("conf") or 0.0)
    if conf >= float(_LABEL_FULL_BOTTLE_MAX_CONF):
        return False
    try:
        _lx1, ly1, _lx2, ly2 = [float(v) for v in label["xyxy"]]
        _bx1, by1, _bx2, by2 = [float(v) for v in bottle["xyxy"]]
    except (TypeError, ValueError, KeyError):
        return False
    bh = by2 - by1
    if bh <= 1e-6:
        return False
    return ((ly2 - ly1) / bh) >= float(_LABEL_FULL_BOTTLE_H_FRAC)


def _best_body_label_on_bottle(
    bottle: dict[str, Any],
    labels: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    """Max conf among body labels that belong to this bottle (not FP silhouette)."""
    best_c = -1.0
    best_lab: dict[str, Any] | None = None
    for lab in labels:
        if _label_in_bottle_ratio(lab, bottle) < float(_LABEL_MIN_IN_BOTTLE_RATIO):
            continue
        if _label_is_full_bottle_false_positive(lab, bottle):
            continue
        frac = _label_center_y_frac(lab, bottle)
        if frac is not None and frac < float(_LABEL_NECK_Y_MAX):
            continue
        conf = float(lab.get("conf") or 0.0)
        if conf > best_c:
            best_c = conf
            best_lab = lab
    return best_c, best_lab


def select_primary_bottle_with_labels(
    bottles: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Главная бутылка с учётом этикеток.

    База — max area. Если на другой бутылке есть сильная этикетка корпуса
    (conf ≥ min и gap над лучшей этикеткой area-primary) — берём её.
    """
    meta: dict[str, Any] = {"mode": "area"}
    by_area = select_primary_bottle(bottles)
    if by_area is None:
        return None, meta
    if len(bottles) <= 1 or not labels:
        return by_area, meta

    scored: list[tuple[float, float, dict[str, Any]]] = []
    for b in bottles:
        best_c, _ = _best_body_label_on_bottle(b, labels)
        scored.append((best_c, float(b.get("area") or 0.0), b))
    # Лучшая по conf этикетки, tie-break площадью
    best_label_bottle = max(scored, key=lambda t: (t[0], t[1]))
    area_best_c, _ = _best_body_label_on_bottle(by_area, labels)
    other = best_label_bottle[2]
    other_c = float(best_label_bottle[0])
    same_as_area = other.get("xyxy") == by_area.get("xyxy")
    if (
        not same_as_area
        and other_c >= float(_PRIMARY_BOTTLE_LABEL_MIN_CONF)
        and (other_c - max(area_best_c, 0.0))
        >= float(_PRIMARY_BOTTLE_LABEL_CONF_GAP)
    ):
        meta = {
            "mode": "best_body_label",
            "area_primary_conf": round(float(by_area.get("conf") or 0.0), 4),
            "area_primary_area": round(float(by_area.get("area") or 0.0), 1),
            "area_primary_best_label_conf": round(max(area_best_c, 0.0), 4)
            if area_best_c >= 0
            else None,
            "chosen_best_label_conf": round(other_c, 4),
            "label_conf_gap": round(other_c - max(area_best_c, 0.0), 4),
        }
        return other, meta
    return by_area, meta


def filter_bottles_by_min_conf(
    bottles: list[dict[str, Any]],
    min_conf: float,
) -> dict[str, Any]:
    """Если бутылок несколько — оставить только conf > порога.

    Одна бутылка не фильтруется. Если ни одна не выше порога — остаются все,
    чтобы слабый кадр всё ещё получил кроп.
    """
    thr = float(min_conf)
    n = len(bottles)
    meta: dict[str, Any] = {
        "threshold": round(thr, 2),
        "input_n": n,
        "kept_n": n,
        "applied": False,
        "reason": "single_or_empty",
    }
    if n <= 1:
        return {"bottles": bottles, "meta": meta}
    strong = [b for b in bottles if float(b.get("conf") or 0.0) > thr]
    if not strong:
        meta["reason"] = "none_above_threshold"
        return {"bottles": bottles, "meta": meta}
    meta["applied"] = True
    meta["kept_n"] = len(strong)
    meta["dropped_n"] = n - len(strong)
    meta["reason"] = "conf_gt_threshold"
    return {"bottles": strong, "meta": meta}


def _label_center_y_frac(
    label: dict[str, Any], bottle: dict[str, Any]
) -> float | None:
    """Относительная высота центра этикетки в рамке бутылки (0=верх, 1=низ)."""
    lx1, ly1, lx2, ly2 = [float(v) for v in label["xyxy"]]
    _bx1, by1, _bx2, by2 = [float(v) for v in bottle["xyxy"]]
    h = by2 - by1
    if h <= 1e-6:
        return None
    cy = 0.5 * (ly1 + ly2)
    return (cy - by1) / h


def _bottle_fallback_crop_box(
    bottle: dict[str, Any],
    *,
    exclude_labels: list[dict[str, Any]] | None = None,
    trim_top: float = _BOTTLE_FALLBACK_TOP_TRIM,
) -> dict[str, Any]:
    """Кроп корпуса бутылки без верхних ценников / чужих рамок."""
    x1, y1, x2, y2 = [float(v) for v in bottle["xyxy"]]
    h = max(0.0, y2 - y1)
    y1b = y1 + float(trim_top) * h
    for lab in exclude_labels or []:
        try:
            _lx1, ly1, _lx2, ly2 = [float(v) for v in lab["xyxy"]]
        except (TypeError, ValueError, KeyError):
            continue
        # Рамка пересекает верх бутылки → обрезаем кроп ниже неё
        if ly2 <= y1 or ly1 >= y2:
            continue
        mid = 0.5 * (ly1 + ly2)
        if mid <= y1 + _LABEL_NECK_Y_MAX * h:
            y1b = max(y1b, ly2)
    y1_cap = y1 + float(_BOTTLE_FALLBACK_TOP_TRIM_MAX) * h
    y1b = min(y1b, y1_cap)
    if y1b >= y2 - 2:
        y1b = y1 + float(trim_top) * h
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1b)
    return {
        "xyxy": [x1, y1b, x2, y2],
        "cls_id": int(bottle.get("cls_id") if bottle.get("cls_id") is not None else -1),
        "cls_name": "bottle_crop",
        "conf": float(bottle.get("conf") or 0.0),
        "area": area,
        "from_bottle": True,
    }


def select_primary_label_with_bottles(
    bottles: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    bottle_min_conf: float = 0.6,
) -> dict[str, Any] | None:
    """Выбор кропа в режиме bottle+label (правила A+B).

    Если бутылок несколько — в выбор идут только те, у которых conf > bottle_min_conf.
    A. Главная бутылка (max area) → среди этикеток на ней с overlap ≥ 0.90
       (≤10% вне контура) и центром в зоне корпуса — обычный select_primary_label.
    B. Если нормальной этикетки нет (нет overlap, только горлышко, или все
       вылезают >10%) → кроп корпуса главной бутылки, без верхних бирок.
       Этикетки на соседних бутылках не берутся вместо fallback.
    Без бутылок — прежнее поведение ``select_primary_label``.
    """
    filtered = filter_bottles_by_min_conf(bottles, bottle_min_conf)
    bottles = list(filtered["bottles"])
    bottle_filter = dict(filtered["meta"])

    if not bottles:
        return select_primary_label(labels)

    primary, primary_meta = select_primary_bottle_with_labels(bottles, labels)
    if primary is None:
        return select_primary_label(labels)

    body: list[dict[str, Any]] = []
    neck: list[dict[str, Any]] = []
    protruding: list[dict[str, Any]] = []
    other_bottle: list[dict[str, Any]] = []

    for lab in labels:
        ratio_pri = _label_in_bottle_ratio(lab, primary)
        best = _best_bottle_ratio(lab, bottles)
        # Вылезает >10% из «своей» лучшей бутылки (или никуда не входит) → не этикетка
        if best < float(_LABEL_MIN_IN_BOTTLE_RATIO):
            drop = dict(lab)
            drop["ignore_reason"] = "protrudes_over_10pct"
            drop["best_bottle_ratio"] = round(best, 4)
            drop["primary_bottle_ratio"] = round(ratio_pri, 4)
            protruding.append(drop)
            continue
        # На другой бутылке целиком — не конкурирует с primary
        if ratio_pri < float(_LABEL_MIN_IN_BOTTLE_RATIO):
            drop = dict(lab)
            drop["ignore_reason"] = "on_other_bottle"
            drop["best_bottle_ratio"] = round(best, 4)
            drop["primary_bottle_ratio"] = round(ratio_pri, 4)
            other_bottle.append(drop)
            continue
        # Weak full-height box on primary → FP silhouette, not label
        if _label_is_full_bottle_false_positive(lab, primary):
            drop = dict(lab)
            drop["ignore_reason"] = "full_bottle_low_conf"
            drop["best_bottle_ratio"] = round(best, 4)
            drop["primary_bottle_ratio"] = round(ratio_pri, 4)
            protruding.append(drop)
            continue
        frac = _label_center_y_frac(lab, primary)
        if frac is None:
            body.append(lab)
            continue
        if frac < float(_LABEL_NECK_Y_MAX):
            drop = dict(lab)
            drop["ignore_reason"] = "neck_band"
            drop["primary_y_frac"] = round(frac, 4)
            neck.append(drop)
            continue
        if float(_LABEL_BODY_Y_MIN) <= frac <= float(_LABEL_BODY_Y_MAX):
            body.append(lab)
        else:
            # ниже корпуса — всё ещё на бутылке, допускаем
            body.append(lab)

    if body:
        pick = select_primary_label(body)
        if pick is None:
            return None
        pick = dict(pick)
        base_reason = str(pick.get("select_reason") or "largest")
        pick["select_reason"] = f"{base_reason}_on_primary_bottle"
        pick["primary_bottle_ratio"] = round(
            _label_in_bottle_ratio(pick, primary), 4
        )
        pick["selection"] = {
            "mode": "label_on_primary_bottle",
            "primary_bottle_area": round(float(primary.get("area") or 0.0), 1),
            "primary_bottle_conf": round(float(primary.get("conf") or 0.0), 4),
            "primary_bottle_pick": primary_meta,
            "body_n": len(body),
            "neck_n": len(neck),
            "protruding_n": len(protruding),
            "other_bottle_n": len(other_bottle),
            "bottle_conf_filter": bottle_filter,
        }
        return pick

    # B: нет нормальной этикетки на главной бутылке → кроп корпуса
    exclude = list(neck) + list(protruding)
    crop = _bottle_fallback_crop_box(primary, exclude_labels=exclude)
    crop["select_reason"] = "bottle_body_fallback"
    crop["selection"] = {
        "mode": "bottle_body_fallback",
        "primary_bottle_area": round(float(primary.get("area") or 0.0), 1),
        "primary_bottle_conf": round(float(primary.get("conf") or 0.0), 4),
        "primary_bottle_pick": primary_meta,
        "bottles_n": len(bottles),
        "bottle_conf_filter": bottle_filter,
        "body_n": 0,
        "neck_n": len(neck),
        "protruding_n": len(protruding),
        "other_bottle_n": len(other_bottle),
        "note": (
            "нет этикетки корпуса на главной бутылке "
            "(overlap≥0.90 и не горлышко) — кроп бутылки без верха"
        ),
    }
    crop["ignored_for_select"] = [
        {
            "ignore_reason": x.get("ignore_reason"),
            "conf": round(float(x.get("conf") or 0.0), 4),
            "area": round(float(x.get("area") or 0.0), 1),
            "best_bottle_ratio": x.get("best_bottle_ratio"),
            "primary_bottle_ratio": x.get("primary_bottle_ratio"),
        }
        for x in (protruding + neck + other_bottle)[:20]
    ]
    return crop


def box_2d_to_xyxy(
    box_2d: list[float] | tuple[float, ...],
    width: int,
    height: int,
) -> list[float]:
    """Gemini/OpenAI [ymin, xmin, ymax, xmax] in 0..1000 → pixel xyxy."""
    if len(box_2d) != 4:
        raise ValueError(f"invalid box_2d: {box_2d}")
    ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
    x1 = max(0.0, min(float(width), xmin / 1000.0 * width))
    y1 = max(0.0, min(float(height), ymin / 1000.0 * height))
    x2 = max(0.0, min(float(width), xmax / 1000.0 * width))
    y2 = max(0.0, min(float(height), ymax / 1000.0 * height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty box after convert: {box_2d}")
    return [x1, y1, x2, y2]


def normalize_box_2d(raw: Any) -> list[int] | None:
    """Validate [ymin,xmin,ymax,xmax] 0..1000 list."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [int(round(float(v))) for v in raw]
    except (TypeError, ValueError):
        return None
    ymin, xmin, ymax, xmax = vals
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        return None
    return vals


def crop_box(image_path: Path, box_xyxy: list[float], out_path: Path) -> Path:
    """Crop xyxy region from image and save."""
    img, _ = open_image_rgb(image_path)
    w, h = img.size
    x1, y1, x2, y2 = box_xyxy
    x1i = max(0, min(w, int(round(x1))))
    y1i = max(0, min(h, int(round(y1))))
    x2i = max(0, min(w, int(round(x2))))
    y2i = max(0, min(h, int(round(y2))))
    if x2i <= x1i or y2i <= y1i:
        raise ValueError(f"invalid crop box: {box_xyxy}")
    crop = img.crop((x1i, y1i, x2i, y2i))
    crop.save(out_path)
    return out_path


def _xyxy_intersection(
    a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]
) -> list[float] | None:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return [ix1, iy1, ix2, iy2]


def _xyxy_center(xyxy: list[float] | tuple[float, ...]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _bottle_key(bottle: dict[str, Any]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bottle["xyxy"]]
    return (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))


def _best_bottle_for_label(
    label: dict[str, Any], bottles: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    if not bottles:
        return None, 0.0
    best_b: dict[str, Any] | None = None
    best_r = -1.0
    for b in bottles:
        r = _label_in_bottle_ratio(label, b)
        if r > best_r:
            best_r = r
            best_b = b
    return best_b, float(best_r) if best_r >= 0 else 0.0


def _primary_main_label(
    raw_labels: list[dict[str, Any]],
    bottles: list[dict[str, Any]],
    primary: dict[str, Any],
) -> dict[str, Any] | None:
    """Крупнейшая raw-этикетка, чья лучшая бутылка = primary."""
    pk = _bottle_key(primary)
    best: dict[str, Any] | None = None
    best_a = -1.0
    for lab in raw_labels:
        b, r = _best_bottle_for_label(lab, bottles)
        if b is None or r < 0.5 or _bottle_key(b) != pk:
            continue
        try:
            a = float(lab.get("area") or _xyxy_area(lab["xyxy"]))
        except (TypeError, ValueError, KeyError):
            continue
        if a > best_a:
            best_a = a
            best = lab
    return best


def _box_fully_inside(
    inner: list[float],
    outer: list[float],
    *,
    eps: float = 1.0,
) -> bool:
    """True если inner целиком внутри outer (с допуском eps px)."""
    ix1, iy1, ix2, iy2 = [float(v) for v in inner]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer]
    return (
        ix1 >= ox1 - eps
        and iy1 >= oy1 - eps
        and ix2 <= ox2 + eps
        and iy2 <= oy2 + eps
    )


# Макс. доля площади основной этикетки, которую может занимать пересечение
# с чужой — иначе это скорее «та же» этикетка / сильный overlap.
_FOREIGN_CUT_MAX_MAIN_OVERLAP = 0.20
# area(inter)/area(foreign) ≥ порога → считаем «полностью внутри» по площади
_FOREIGN_CUT_FULL_INSIDE_FRAC = 0.98
# Не вырезать кусок > этой доли selected-кропа (иначе «заливка всего label»)
_FOREIGN_CUT_MAX_SELECTED_FRAC = 0.35


def _xyxy_close(
    a: list[float],
    b: list[float],
    *,
    tol: float = 3.0,
) -> bool:
    if len(a) != 4 or len(b) != 4:
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def foreign_label_cut_rects(
    selected_xyxy: list[float],
    bottles: list[dict[str, Any]],
    raw_labels: list[dict[str, Any]],
    *,
    primary_bottle: dict[str, Any] | None = None,
    max_main_overlap: float = _FOREIGN_CUT_MAX_MAIN_OVERLAP,
) -> list[dict[str, Any]]:
    """Прямоугольный вырез чужой этикетки из selected-кропа.

    Условия (все):
    - raw-этикетка принадлежит другой бутылке (best ≠ primary);
    - не совпадает с selected (сама выбранная этикетка);
    - пересекается с основной (main = selected crop);
    - чужая **не** целиком внутри main (ни по боксу, ни по площади ≥98%);
    - area(foreign∩main) / area(main) ≤ max_main_overlap (по умолчанию 20%);
    - вырез ≤ 35% площади selected (защита от залития всего кропа);
    - есть пересечение с selected (что реально вырезаем на кропе).
    Треугольники / наклон границы — отключены.
    """
    if not bottles or not raw_labels or not selected_xyxy:
        return []
    try:
        sel_xy = [float(v) for v in selected_xyxy]
    except (TypeError, ValueError):
        return []
    sel_area = _xyxy_area(sel_xy)
    if sel_area <= 0:
        return []

    # Primary = бутылка выбранного кропа (не max-area — иначе cut «съедает» label).
    owner, owner_r = _best_bottle_for_label({"xyxy": sel_xy}, bottles)
    if owner is not None and owner_r >= 0.5:
        primary = owner
    else:
        primary = primary_bottle or select_primary_bottle(bottles)
    if primary is None:
        return []
    pk = _bottle_key(primary)

    # Main = сам selected crop (то, что реально идёт в OCR/embed).
    main_xy = sel_xy
    main_area = sel_area

    cuts: list[dict[str, Any]] = []
    for lab in raw_labels:
        try:
            lxy = [float(v) for v in lab["xyxy"]]
        except (TypeError, ValueError, KeyError):
            continue
        if _xyxy_close(lxy, sel_xy):
            continue
        best_b, best_r = _best_bottle_for_label(lab, bottles)
        if best_b is None or best_r < 0.5:
            continue
        if _bottle_key(best_b) == pk:
            continue
        foreign_area = float(lab.get("area") or _xyxy_area(lxy))
        if foreign_area <= 0:
            continue
        inter_main = _xyxy_intersection(lxy, main_xy)
        if inter_main is None:
            continue
        inter_main_area = _xyxy_area(inter_main)
        if inter_main_area <= 0:
            continue
        overlap_main = inter_main_area / main_area
        overlap_foreign = inter_main_area / foreign_area
        if _box_fully_inside(lxy, main_xy):
            continue
        if overlap_foreign >= float(_FOREIGN_CUT_FULL_INSIDE_FRAC):
            continue
        if overlap_main > float(max_main_overlap):
            continue
        # Что заливаем на кропе — пересечение чужой с selected box
        inter_sel = _xyxy_intersection(lxy, sel_xy)
        if inter_sel is None:
            continue
        ia_sel = _xyxy_area(inter_sel)
        sx1, sy1, sx2, sy2 = inter_sel
        if ia_sel < 400.0 or (sx2 - sx1) < 4.0 or (sy2 - sy1) < 4.0:
            continue
        if (ia_sel / sel_area) > float(_FOREIGN_CUT_MAX_SELECTED_FRAC):
            continue
        cuts.append(
            {
                "rect_xyxy": [round(v, 2) for v in inter_sel],
                "inter_main_xyxy": [round(v, 2) for v in inter_main],
                "foreign_xyxy": [round(v, 2) for v in lxy],
                "foreign_conf": round(float(lab.get("conf") or 0.0), 4),
                "best_bottle_ratio": round(best_r, 4),
                "overlap_main": round(overlap_main, 4),
                "overlap_foreign": round(overlap_foreign, 4),
                "inter_sel_area": round(ia_sel, 1),
                "main_xyxy": [round(v, 2) for v in main_xy],
                "reason": "partial_foreign_overlap_le_20pct_main",
            }
        )
    return cuts


# Обратная совместимость имени (треугольники отключены)
def foreign_label_cut_triangles(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    kwargs.pop("gray", None)
    kwargs.pop("min_inter_frac", None)
    kwargs.pop("min_inter_area", None)
    return foreign_label_cut_rects(*args, **kwargs)


def crop_box_mask_foreign_labels(
    image_path: Path,
    box_xyxy: list[float],
    out_path: Path,
    *,
    bottles: list[dict[str, Any]] | None = None,
    raw_labels: list[dict[str, Any]] | None = None,
    primary_bottle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Кроп selected label + заливка прямоугольников чужих этикеток.

    Заливка — медиана центра кропа. Треугольники отключены.
    """
    cuts = foreign_label_cut_rects(
        box_xyxy,
        list(bottles or []),
        list(raw_labels or []),
        primary_bottle=primary_bottle,
    )
    img, _ = open_image_rgb(image_path)
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1i = max(0, min(w, int(round(x1))))
    y1i = max(0, min(h, int(round(y1))))
    x2i = max(0, min(w, int(round(x2))))
    y2i = max(0, min(h, int(round(y2))))
    if x2i <= x1i or y2i <= y1i:
        raise ValueError(f"invalid crop box: {box_xyxy}")
    crop = img.crop((x1i, y1i, x2i, y2i))
    meta: dict[str, Any] = {
        "ok": True,
        "mode": "rect_partial_overlap",
        "max_main_overlap": _FOREIGN_CUT_MAX_MAIN_OVERLAP,
        "cuts_n": 0,
        "cuts": [],
        "fill_rgb": None,
    }
    if not cuts:
        crop.save(out_path)
        return meta

    arr = np.asarray(crop)
    ch, cw = arr.shape[:2]
    cy0, cy1 = max(0, ch // 4), min(ch, (3 * ch) // 4)
    cx0, cx1 = max(0, cw // 4), min(cw, (3 * cw) // 4)
    if cy1 > cy0 and cx1 > cx0:
        med = np.median(arr[cy0:cy1, cx0:cx1], axis=(0, 1))
        fill = tuple(int(max(0, min(255, round(float(v))))) for v in med[:3])
    else:
        fill = (245, 245, 245)
    draw = ImageDraw.Draw(crop)
    local_cuts: list[dict[str, Any]] = []
    for cut in cuts:
        rx1, ry1, rx2, ry2 = [float(v) for v in cut["rect_xyxy"]]
        loc = [
            max(0.0, min(float(cw), rx1 - x1i)),
            max(0.0, min(float(ch), ry1 - y1i)),
            max(0.0, min(float(cw), rx2 - x1i)),
            max(0.0, min(float(ch), ry2 - y1i)),
        ]
        if loc[2] - loc[0] < 1.0 or loc[3] - loc[1] < 1.0:
            continue
        draw.rectangle(loc, fill=fill)
        local = dict(cut)
        local["rect_local"] = [round(v, 2) for v in loc]
        local_cuts.append(local)
    crop.save(out_path)
    meta["cuts_n"] = len(local_cuts)
    meta["cuts"] = local_cuts
    meta["fill_rgb"] = list(fill)
    return meta
