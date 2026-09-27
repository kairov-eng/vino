"""DeepSeek vision OCR for findwine (OpenAI-compatible API).

Env:
  DEEPSEEK_API_KEY (or DEEPSEEK_KEY)
  DEEPSEEK_BASE_URL=https://api.deepseek.com
  DEEPSEEK_MODEL=deepseek-flash

Prompt: OCR_PROMPT_PATH (общий с Gemini / OpenAI).
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
from app.pipeline.label_detect_llm import load_ocr_prompt, text_from_ocr_fields
from app.pipeline.ocr_gemini import _prepare_jpeg_bytes
from app.pipeline.ocr_match import normalize_ocr_text

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_json_payload(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("пустой ответ DeepSeek")
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
        raise ValueError("DeepSeek JSON не объект")
    return data


def _call_deepseek_vision(
    *,
    api_key: str,
    image_bytes: bytes,
    prompt: str,
    model: str,
    timeout: float,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    t0 = time.perf_counter()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    base = (
        getattr(app_config, "DEEPSEEK_BASE_URL", "") or "https://api.deepseek.com"
    ).strip().rstrip("/")
    # OpenAI SDK base_url=https://api.deepseek.com → /chat/completions
    # (без лишнего /v1; /v1 тоже принимается DeepSeek)
    url = f"{base}/chat/completions"
    payload: dict[str, Any] = {
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
        raise RuntimeError(f"DeepSeek HTTP {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("DeepSeek: пустой choices")
    message = (choices[0] or {}).get("message") or {}
    raw_text = str(message.get("content") or "").strip()
    parsed = _parse_json_payload(raw_text)
    usage = body.get("usage") or {}
    meta = {
        "provider": "deepseek",
        "base_url": base,
        "tokens_input": usage.get("prompt_tokens"),
        "tokens_output": usage.get("completion_tokens"),
        "tokens_total": usage.get("total_tokens"),
        "deepseek_model": body.get("model") or model,
        "wall_ms": round(wall_ms, 1),
        "finish_reason": (choices[0] or {}).get("finish_reason"),
    }
    return parsed, wall_ms, meta


def run_deepseek_ocr(label_path: Path) -> dict[str, Any]:
    """OCR-only on label crop via DeepSeek (OpenAI-compatible)."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "id": "deepseek",
        "engine": "deepseek",
        "engine_name": "DeepSeek",
        "preprocess": "A",
        "preprocess_name": "original",
        "name": "original+DeepSeek",
        "ok": False,
        "prompt": "ocr_only",
    }
    if not getattr(app_config, "USE_DEEPSEEK_OCR", True):
        out.update(
            {"skipped": True, "error": "DeepSeek OCR выключен (USE_DEEPSEEK_OCR)"}
        )
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    api_key = (
        getattr(app_config, "DEEPSEEK_API_KEY", "")
        or getattr(app_config, "DEEPSEEK_KEY", "")
        or ""
    ).strip()
    if not api_key:
        out.update(
            {"skipped": True, "error": "DEEPSEEK_API_KEY не задан"}
        )
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    try:
        jpeg, prep_meta = _prepare_jpeg_bytes(label_path)
        out["image_prep"] = prep_meta
        model = (
            getattr(app_config, "DEEPSEEK_MODEL", "") or ""
        ).strip() or "deepseek-flash"
        out["model"] = model
        timeout = float(
            getattr(app_config, "DEEPSEEK_OCR_TIMEOUT_SEC", None) or 120
        )
        parsed, api_ms, api_meta = _call_deepseek_vision(
            api_key=api_key,
            image_bytes=jpeg,
            prompt=load_ocr_prompt(),
            model=model,
            timeout=timeout,
        )
        out["api_ms"] = round(api_ms, 1)
        out["model"] = api_meta.get("deepseek_model") or model
        out["api"] = api_meta
        tokens_in = api_meta.get("tokens_input")
        tokens_out = api_meta.get("tokens_output")
        out["tokens_input"] = tokens_in
        out["tokens_output"] = tokens_out

        full, lines = text_from_ocr_fields(parsed)
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
            out["error"] = "DeepSeek вернул пустой full_text"
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
