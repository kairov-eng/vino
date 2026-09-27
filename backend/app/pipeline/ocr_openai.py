"""OpenAI vision OCR (+ optional label-box detect) for findwine.

Prompts (общие с Gemini, пути из .env):
  - OCR_PROMPT_PATH — OCR only (кроп уже есть)
  - LABEL_OCR_PROMPT_PATH — рамка + OCR на полный кадр

Env: OPENAI_KEY (or OPENAI_API_KEY), OPENAI_MODEL.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from app.db import config as app_config
from app.pipeline.label_detect_llm import (
    finalize_detect_result,
    load_label_ocr_prompt,
    load_ocr_prompt,
    resolve_image_sizes,
    text_from_ocr_fields,
)
from app.pipeline.ocr_gemini import _prepare_jpeg_bytes
from app.pipeline.ocr_match import normalize_ocr_text

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_json_payload(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("пустой ответ OpenAI")
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("OpenAI JSON не объект")
    return data


def _call_openai_vision(
    *,
    api_key: str,
    image_bytes: bytes,
    prompt: str,
    model: str,
    timeout: float,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    t0 = time.perf_counter()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    url = (app_config.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
    url = f"{url}/chat/completions"
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        },
                    },
                ],
            }
        ],
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI: пустой choices")
    message = (choices[0] or {}).get("message") or {}
    raw_text = str(message.get("content") or "").strip()
    parsed = _parse_json_payload(raw_text)
    usage = body.get("usage") or {}
    meta = {
        "tokens_input": usage.get("prompt_tokens"),
        "tokens_output": usage.get("completion_tokens"),
        "tokens_total": usage.get("total_tokens"),
        "openai_model": body.get("model") or model,
        "wall_ms": round(wall_ms, 1),
        "finish_reason": (choices[0] or {}).get("finish_reason"),
    }
    return parsed, wall_ms, meta


def _text_from_ocr_fields(parsed: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    return text_from_ocr_fields(parsed)


def run_openai_ocr(label_path: Path) -> dict[str, Any]:
    """OCR label via OpenAI vision (OCR-only prompt)."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "id": "openai",
        "engine": "openai",
        "engine_name": "OpenAI",
        "preprocess": "A",
        "preprocess_name": "original",
        "name": "original+OpenAI",
        "ok": False,
        "prompt": "ocr_only",
    }
    if not app_config.USE_OPENAI_OCR:
        out.update({"skipped": True, "error": "OpenAI OCR выключен (USE_OPENAI_OCR)"})
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    api_key = (app_config.OPENAI_KEY or "").strip()
    if not api_key:
        out.update({"skipped": True, "error": "OPENAI_KEY не задан"})
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    try:
        jpeg, prep_meta = _prepare_jpeg_bytes(label_path)
        out["image_prep"] = prep_meta
        model = (app_config.OPENAI_MODEL or "").strip() or "gpt-4.1-mini"
        out["model"] = model
        parsed, api_ms, api_meta = _call_openai_vision(
            api_key=api_key,
            image_bytes=jpeg,
            prompt=load_ocr_prompt(),
            model=model,
            timeout=float(app_config.OPENAI_OCR_TIMEOUT_SEC),
        )
        out["api_ms"] = round(api_ms, 1)
        out["model"] = api_meta.get("openai_model") or model
        out["api"] = api_meta
        tokens_in = api_meta.get("tokens_input")
        tokens_out = api_meta.get("tokens_output")
        out["tokens_input"] = tokens_in
        out["tokens_output"] = tokens_out

        full, lines = _text_from_ocr_fields(parsed)
        json_payload = dict(parsed)
        json_payload["tokens_input"] = tokens_in
        json_payload["tokens_output"] = tokens_out
        out.update(
            {
                "ok": bool(full.strip()),
                "text": full,
                "text_norm": normalize_ocr_text(full),
                "lines": lines,
                "json": json_payload,
                "ocr_ms": round(api_ms, 1),
            }
        )
        if not out["ok"]:
            out["error"] = "OpenAI вернул пустой full_text"
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def run_openai_label_detect_ocr(
    image_path: Path,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Detect label boxes + OCR on full frame (replaces YOLO when use_yolo=false)."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "id": "openai_label_detect",
        "engine": "openai",
        "engine_name": "OpenAI label+OCR",
        "ok": False,
        "prompt": "label_ocr",
        "boxes": [],
        "selected": None,
    }
    api_key = (app_config.OPENAI_KEY or "").strip()
    if not api_key:
        out.update({"skipped": True, "error": "OPENAI_KEY не задан"})
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    try:
        jpeg, prep_meta = _prepare_jpeg_bytes(image_path)
        out["image_prep"] = prep_meta
        sent_size, orig_size = resolve_image_sizes(image_path, prep_meta, image_size)

        model = (app_config.OPENAI_MODEL or "").strip() or "gpt-4.1-mini"
        out["model"] = model
        parsed, api_ms, api_meta = _call_openai_vision(
            api_key=api_key,
            image_bytes=jpeg,
            prompt=load_label_ocr_prompt(),
            model=model,
            timeout=float(app_config.OPENAI_OCR_TIMEOUT_SEC),
        )
        out.update(
            finalize_detect_result(
                engine="openai",
                engine_name="OpenAI",
                parsed=parsed,
                api_ms=api_ms,
                tokens_in=api_meta.get("tokens_input"),
                tokens_out=api_meta.get("tokens_output"),
                model=api_meta.get("openai_model") or model,
                api_meta=api_meta,
                sent_size=sent_size,
                orig_size=orig_size,
            )
        )
        out["api"] = api_meta
        out["model"] = out.get("model") or model
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
