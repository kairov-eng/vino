"""Shared helpers for LLM label-box detect + OCR (Gemini / OpenAI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.db import config as app_config
from app.pipeline.ocr_match import normalize_ocr_text
from app.pipeline.yolo_detect import (
    box_2d_to_xyxy,
    normalize_box_2d,
    select_primary_label,
)

# Legacy filenames if env/default file missing
_LEGACY_LABEL_OCR = (
    Path(__file__).with_name("openai_label_ocr_prompt.txt"),
    Path(__file__).with_name("_label_ocr_prompt.txt"),
)
_LEGACY_OCR = (
    Path(__file__).with_name("gemini_ocr_prompt.txt"),
)


def _resolve_existing_prompt(primary: Path, legacy: tuple[Path, ...]) -> Path:
    if primary.is_file():
        return primary
    for alt in legacy:
        if alt.is_file():
            return alt
    return primary


def load_label_ocr_prompt() -> str:
    """Рамка+OCR (полный кадр) — общий для Gemini и OpenAI."""
    path = _resolve_existing_prompt(
        Path(app_config.LABEL_OCR_PROMPT_PATH),
        _LEGACY_LABEL_OCR,
    )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Пустой промпт label+OCR: {path}")
    return text


def load_ocr_prompt() -> str:
    """OCR-only по кропу этикетки — общий для Gemini и OpenAI."""
    path = _resolve_existing_prompt(
        Path(app_config.OCR_PROMPT_PATH),
        _LEGACY_OCR,
    )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Пустой промпт OCR: {path}")
    return text


def text_from_ocr_fields(parsed: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    full = str(parsed.get("full_text") or parsed.get("label_text") or "").strip()
    if not full and isinstance(parsed.get("lines"), list):
        full = "\n".join(str(x).strip() for x in parsed["lines"] if str(x).strip())
    lines: list[dict[str, Any]] = []
    if isinstance(parsed.get("lines"), list):
        lines = [
            {"text": str(x).strip(), "score": None, "box": None}
            for x in parsed["lines"]
            if str(x).strip()
        ]
    elif full:
        lines = [
            {"text": ln, "score": None, "box": None}
            for ln in full.splitlines()
            if ln.strip()
        ]
    return full, lines


def parse_label_items(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize detect JSON to list of items with validated label_box_2d."""
    items_raw = parsed.get("items")
    items: list[dict[str, Any]] = []
    if isinstance(items_raw, list):
        for it in items_raw:
            if isinstance(it, dict):
                items.append(it)
    elif normalize_box_2d(parsed.get("label_box_2d")) is not None:
        items.append(parsed)

    out: list[dict[str, Any]] = []
    for it in items:
        box = normalize_box_2d(it.get("label_box_2d"))
        if box is None:
            continue
        ymin, xmin, ymax, xmax = box
        area_norm = max(0, ymax - ymin) * max(0, xmax - xmin)
        out.append({**it, "label_box_2d": box, "area_norm": area_norm})
    return out


_OCR_JSON_KEYS = (
    "producer",
    "wine_name",
    "color",
    "type",
    "cupage",
    "vintage_year",
    "foundation_year",
    "region",
    "alcohol",
    "volume",
    "other",
    "full_text",
    "lines",
)


def boxes_from_parsed(
    parsed: dict[str, Any],
    *,
    sent_size: tuple[int, int],
    orig_size: tuple[int, int],
    source: str,
) -> list[dict[str, Any]]:
    """Convert LLM JSON → pixel boxes on original image (+ OCR fields)."""
    rw, rh = sent_size
    orig_w, orig_h = orig_size
    scale_x = float(orig_w) / float(rw)
    scale_y = float(orig_h) / float(rh)
    boxes: list[dict[str, Any]] = []
    for it in parse_label_items(parsed):
        xyxy_sent = box_2d_to_xyxy(it["label_box_2d"], int(rw), int(rh))
        xyxy = [
            xyxy_sent[0] * scale_x,
            xyxy_sent[1] * scale_y,
            xyxy_sent[2] * scale_x,
            xyxy_sent[3] * scale_y,
        ]
        area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
        full, lines = text_from_ocr_fields(it)
        boxes.append(
            {
                "xyxy": xyxy,
                "box_2d": it["label_box_2d"],
                "conf": 1.0,
                "cls_name": "label",
                "area": area,
                "source": source,
                "bottle_box_2d": normalize_box_2d(it.get("bottle_box_2d")),
                "text": full,
                "text_norm": normalize_ocr_text(full),
                "lines": lines,
                "ocr_json": {k: it.get(k) for k in _OCR_JSON_KEYS if k in it},
            }
        )
    return boxes


def resolve_image_sizes(
    image_path: Path,
    prep_meta: dict[str, Any],
    image_size: tuple[int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ((sent_w, sent_h), (orig_w, orig_h))."""
    rw, rh = prep_meta.get("resized_size") or [None, None]
    if not rw or not rh:
        with Image.open(image_path) as im:
            rw, rh = im.size
    if image_size:
        orig_w, orig_h = image_size
    else:
        ow = (prep_meta.get("original_size") or [None, None])[0]
        oh = (prep_meta.get("original_size") or [None, None])[1]
        if ow and oh:
            orig_w, orig_h = int(ow), int(oh)
        else:
            with Image.open(image_path) as im:
                orig_w, orig_h = im.size
    return (int(rw), int(rh)), (int(orig_w), int(orig_h))


def build_ocr_variant_from_selected(
    *,
    engine: str,
    engine_name: str,
    selected: dict[str, Any],
    api_ms: float,
    tokens_in: Any,
    tokens_out: Any,
    model: str | None,
    api_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Status-compatible OCR variant (gemini/openai) from detect+OCR result."""
    text = selected.get("text") or ""
    variant = {
        "id": engine,
        "engine": engine,
        "engine_name": engine_name,
        "preprocess": "A",
        "preprocess_name": "original",
        "name": f"full_frame+{engine_name} label+OCR",
        "ok": bool(str(text).strip()),
        "prompt": "label_ocr",
        "from_label_detect": True,
        "text": text,
        "text_norm": selected.get("text_norm") or "",
        "lines": selected.get("lines") or [],
        "json": selected.get("ocr_json") or {},
        "ocr_ms": round(api_ms, 1),
        "ms": round(api_ms, 1),
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "model": model,
        "select_reason": selected.get("select_reason"),
    }
    if api_meta:
        if engine == "gemini":
            variant["proxy"] = api_meta
        else:
            variant["api"] = api_meta
    if not variant["ok"]:
        variant["error"] = f"{engine_name}: пустой текст выбранной этикетки"
    return variant


def finalize_detect_result(
    *,
    engine: str,
    engine_name: str,
    parsed: dict[str, Any],
    api_ms: float,
    tokens_in: Any,
    tokens_out: Any,
    model: str | None,
    api_meta: dict[str, Any] | None,
    sent_size: tuple[int, int],
    orig_size: tuple[int, int],
) -> dict[str, Any]:
    """Common post-processing after LLM label+OCR JSON."""
    boxes = boxes_from_parsed(
        parsed, sent_size=sent_size, orig_size=orig_size, source=engine
    )
    selected = select_primary_label(boxes)
    out: dict[str, Any] = {
        "ok": bool(selected),
        "boxes": boxes,
        "boxes_count": len(boxes),
        "selected": selected,
        "json": parsed,
        "api_ms": round(api_ms, 1),
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "model": model,
        "prompt": "label_ocr",
    }
    if selected:
        out["text"] = selected.get("text") or ""
        out["text_norm"] = selected.get("text_norm") or ""
        out["lines"] = selected.get("lines") or []
        out["ocr_variant"] = build_ocr_variant_from_selected(
            engine=engine,
            engine_name=engine_name,
            selected=selected,
            api_ms=api_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            api_meta=api_meta,
        )
    else:
        out["error"] = f"{engine_name} не нашёл этикетку (label_box_2d)"
    return out
