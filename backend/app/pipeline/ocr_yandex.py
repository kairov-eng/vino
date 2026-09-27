"""Yandex Vision OCR for findwine (recognizeText API).

Env:
  YANDEX_API_KEY
  YANDEX_FOLDER_ID
  YANDEX_OCR_URL (default recognizeText)
  YANDEX_OCR_MODEL=page
  YANDEX_OCR_LANGUAGES=*
  YANDEX_OCR_TIMEOUT_SEC

Для пайплайна нужен только массив распознанных строк (lines).
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import requests

from app.db import config as app_config
from app.pipeline.label_detect_llm import text_from_ocr_fields
from app.pipeline.ocr_gemini import _prepare_jpeg_bytes
from app.pipeline.ocr_match import normalize_ocr_text

DEFAULT_OCR_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
DEFAULT_MODEL = "page"
DEFAULT_LANGS = "*"


def _extract_lines(body: dict[str, Any]) -> list[str]:
    """Строки из ответа Yandex OCR (blocks → lines; fallback fullText)."""
    lines_out: list[str] = []
    try:
        result = body.get("result") or {}
        if isinstance(result, list):
            annotations = result
        else:
            ann = result.get("textAnnotation") or result
            annotations = [ann] if isinstance(ann, dict) else []

        full_fallback = ""
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            full = ann.get("fullText") or ann.get("text")
            if isinstance(full, str) and full.strip() and not full_fallback:
                full_fallback = full.strip()
            for block in ann.get("blocks") or []:
                if not isinstance(block, dict):
                    continue
                for line in block.get("lines") or []:
                    if not isinstance(line, dict):
                        continue
                    t = line.get("text")
                    if isinstance(t, str) and t.strip():
                        lines_out.append(t.strip())
                        continue
                    words = line.get("words") or []
                    wtxt = " ".join(
                        str(w.get("text") or "").strip()
                        for w in words
                        if isinstance(w, dict) and str(w.get("text") or "").strip()
                    )
                    if wtxt:
                        lines_out.append(wtxt)
        if not lines_out and full_fallback:
            lines_out = [ln.strip() for ln in full_fallback.splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        pass
    return lines_out


def _call_yandex_ocr(
    *,
    image_bytes: bytes,
    api_key: str,
    folder_id: str,
    url: str,
    model: str,
    language_codes: list[str],
    timeout: float,
) -> tuple[dict[str, Any], float, list[str]]:
    payload = {
        "mimeType": "JPEG",
        "languageCodes": language_codes,
        "model": model,
        "content": base64.b64encode(image_bytes).decode("utf-8"),
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
        "x-data-logging-enabled": "true",
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    wall_ms = (time.perf_counter() - t0) * 1000
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = {"_raw": (resp.text or "")[:2000], "_http_status": resp.status_code}
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Yandex OCR HTTP {resp.status_code}: {str(body)[:500]}"
        )
    body_dict = body if isinstance(body, dict) else {"result": body}
    lines = _extract_lines(body_dict)
    return body_dict, wall_ms, lines


def run_yandex_ocr(label_path: Path) -> dict[str, Any]:
    """OCR-only on label crop via Yandex Vision → lines[]."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "id": "yandex",
        "engine": "yandex",
        "engine_name": "Yandex OCR",
        "preprocess": "A",
        "preprocess_name": "original",
        "name": "original+Yandex",
        "ok": False,
        "prompt": "ocr_lines_only",
    }
    if not getattr(app_config, "USE_YANDEX_OCR", True):
        out.update(
            {"skipped": True, "error": "Yandex OCR выключен (USE_YANDEX_OCR)"}
        )
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    api_key = (getattr(app_config, "YANDEX_API_KEY", "") or "").strip()
    folder_id = (getattr(app_config, "YANDEX_FOLDER_ID", "") or "").strip()
    if not api_key:
        out.update({"skipped": True, "error": "YANDEX_API_KEY не задан"})
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    if not folder_id:
        out.update({"skipped": True, "error": "YANDEX_FOLDER_ID не задан"})
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    try:
        jpeg, prep_meta = _prepare_jpeg_bytes(label_path)
        out["image_prep"] = prep_meta
        url = (
            getattr(app_config, "YANDEX_OCR_URL", "") or DEFAULT_OCR_URL
        ).strip() or DEFAULT_OCR_URL
        model = (
            getattr(app_config, "YANDEX_OCR_MODEL", "") or DEFAULT_MODEL
        ).strip() or DEFAULT_MODEL
        langs_raw = (
            getattr(app_config, "YANDEX_OCR_LANGUAGES", "") or DEFAULT_LANGS
        ).strip() or DEFAULT_LANGS
        language_codes = [x.strip() for x in langs_raw.split(",") if x.strip()] or [
            "*"
        ]
        timeout = float(
            getattr(app_config, "YANDEX_OCR_TIMEOUT_SEC", None) or 120
        )
        out["model"] = model
        body, api_ms, line_strings = _call_yandex_ocr(
            image_bytes=jpeg,
            api_key=api_key,
            folder_id=folder_id,
            url=url,
            model=model,
            language_codes=language_codes,
            timeout=timeout,
        )
        out["api_ms"] = round(api_ms, 1)
        out["api"] = {
            "provider": "yandex",
            "url": url,
            "model": model,
            "languages": language_codes,
            "wall_ms": round(api_ms, 1),
        }
        full, lines = text_from_ocr_fields({"lines": line_strings})
        out.update(
            {
                "ok": bool(full.strip()),
                "text": full,
                "text_norm": normalize_ocr_text(full),
                "lines": lines,
                # Сырой ответ большой — в status только строки
                "json": {"lines": line_strings},
                "ocr_ms": round(api_ms, 1),
                "raw_keys": sorted(body.keys()) if isinstance(body, dict) else [],
            }
        )
        if not out["ok"]:
            out["error"] = "Yandex OCR вернул пустой текст"
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
