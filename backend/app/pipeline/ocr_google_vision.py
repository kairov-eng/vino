"""Google Cloud Vision OCR for findwine (как label_detect/13.probe_google_vision_ocr.py).

Env:
  USE_GOOGLE_VISION_OCR
  GOOGLE_APPLICATION_CREDENTIALS  (тот же SA, что Vertex)
  GOOGLE_VISION_FEATURE=DOCUMENT_TEXT_DETECTION | TEXT_DETECTION
  GOOGLE_VISION_LANGUAGES=ru,en   (опционально, language_hints)
  GOOGLE_VISION_TIMEOUT_SEC

Для пайплайна — массив строк (lines) + full text.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from app.db import config as app_config
from app.pipeline.gemini_client import apply_application_credentials
from app.pipeline.label_detect_llm import text_from_ocr_fields
from app.pipeline.ocr_gemini import _prepare_jpeg_bytes
from app.pipeline.ocr_match import normalize_ocr_text

DEFAULT_FEATURE = "DOCUMENT_TEXT_DETECTION"


def _extract_lines_from_full(full: str) -> list[str]:
    return [ln.strip() for ln in (full or "").splitlines() if ln.strip()]


def _extract_lines_from_document(annotation: Any) -> list[str]:
    """Параграфы / строки из full_text_annotation (если есть структура)."""
    lines_out: list[str] = []
    try:
        pages = getattr(annotation, "pages", None) or []
        for page in pages:
            for block in getattr(page, "blocks", None) or []:
                for para in getattr(block, "paragraphs", None) or []:
                    parts: list[str] = []
                    for word in getattr(para, "words", None) or []:
                        syms = "".join(
                            str(getattr(s, "text", "") or "")
                            for s in (getattr(word, "symbols", None) or [])
                        )
                        if syms.strip():
                            parts.append(syms.strip())
                    if parts:
                        lines_out.append(" ".join(parts))
    except Exception:  # noqa: BLE001
        pass
    return lines_out


def _call_google_vision(
    *,
    image_bytes: bytes,
    feature: str,
    language_hints: list[str],
) -> tuple[str, list[str], float, dict[str, Any]]:
    from google.cloud import vision

    apply_application_credentials()
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    image_context = None
    if language_hints:
        image_context = vision.ImageContext(language_hints=language_hints)

    feat = (feature or DEFAULT_FEATURE).upper().strip()
    t0 = time.perf_counter()
    if feat in {"TEXT_DETECTION", "TEXT", "TEXT_DETECT"}:
        response = client.text_detection(image=image, image_context=image_context)
        if response.error.message:
            raise RuntimeError(response.error.message)
        texts = response.text_annotations
        full = (texts[0].description if texts else "") or ""
        line_strings = _extract_lines_from_full(full)
    elif feat in {
        "DOCUMENT_TEXT_DETECTION",
        "DOCUMENT",
        "DOCUMENT_TEXT",
        "DOC",
    }:
        response = client.document_text_detection(
            image=image, image_context=image_context
        )
        if response.error.message:
            raise RuntimeError(response.error.message)
        full = ""
        ann = response.full_text_annotation
        if ann and ann.text:
            full = ann.text
        elif response.text_annotations:
            full = response.text_annotations[0].description or ""
        line_strings = _extract_lines_from_document(ann) if ann else []
        if not line_strings:
            line_strings = _extract_lines_from_full(full)
    else:
        raise RuntimeError(
            f"Неизвестный GOOGLE_VISION_FEATURE={feature!r} "
            "(TEXT_DETECTION | DOCUMENT_TEXT_DETECTION)"
        )
    api_ms = (time.perf_counter() - t0) * 1000
    meta = {
        "provider": "google_vision",
        "feature": feat,
        "language_hints": language_hints,
        "wall_ms": round(api_ms, 1),
        "project": (getattr(app_config, "GOOGLE_CLOUD_PROJECT", "") or "") or None,
    }
    return full.strip(), line_strings, api_ms, meta


def run_google_vision_ocr(
    label_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """OCR-only on label crop via Google Cloud Vision → lines[].

    force=True — вызвать даже если USE_GOOGLE_VISION_OCR выключен
    (fallback после падения OpenAI/Gemini).
    """
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "id": "google_vision",
        "engine": "google_vision",
        "engine_name": "Google Vision OCR",
        "preprocess": "A",
        "preprocess_name": "original",
        "name": "original+GoogleVision",
        "ok": False,
        "prompt": "ocr_lines_only",
    }
    if not force and not getattr(app_config, "USE_GOOGLE_VISION_OCR", True):
        out.update(
            {
                "skipped": True,
                "error": "Google Vision OCR выключен (USE_GOOGLE_VISION_OCR)",
            }
        )
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    cred = (getattr(app_config, "GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    if not cred and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        out.update(
            {
                "skipped": True,
                "error": "GOOGLE_APPLICATION_CREDENTIALS не задан",
            }
        )
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    try:
        jpeg, prep_meta = _prepare_jpeg_bytes(label_path)
        out["image_prep"] = prep_meta
        feature = (
            getattr(app_config, "GOOGLE_VISION_FEATURE", "") or DEFAULT_FEATURE
        ).strip() or DEFAULT_FEATURE
        langs_raw = (
            getattr(app_config, "GOOGLE_VISION_LANGUAGES", "") or ""
        ).strip()
        language_hints = [x.strip() for x in langs_raw.split(",") if x.strip()]
        out["model"] = feature
        full, line_strings, api_ms, api_meta = _call_google_vision(
            image_bytes=jpeg,
            feature=feature,
            language_hints=language_hints,
        )
        out["api_ms"] = round(api_ms, 1)
        out["api"] = api_meta
        # Prefer structured lines; if empty, split full
        if not line_strings and full:
            line_strings = _extract_lines_from_full(full)
        text_full, lines = text_from_ocr_fields(
            {"full_text": full, "lines": line_strings}
        )
        out.update(
            {
                "ok": bool(text_full.strip()),
                "text": text_full,
                "text_norm": normalize_ocr_text(text_full),
                "lines": lines,
                "json": {"lines": line_strings, "full_text": text_full},
                "ocr_ms": round(api_ms, 1),
            }
        )
        if not out["ok"]:
            out["error"] = "Google Vision OCR вернул пустой текст"
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
