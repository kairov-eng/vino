"""Hugging Face Qwen2.5-VL OCR for findwine (Inference Endpoint).

OpenAI-compatible: POST {endpoint}/v1/chat/completions
Image as data:image/jpeg;base64,… in messages[].content[].image_url.

Prompt: OCR_PROMPT_PATH (тот же, что Gemini / OpenAI OCR-only).
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
        raise ValueError("пустой ответ Qwen OCR")
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
        raise ValueError("Qwen OCR JSON не объект")
    return data


def _hf_token() -> str:
    return (
        getattr(app_config, "HF_ACCESS_TOKEN", "")
        or getattr(app_config, "HF_TOKEN", "")
        or ""
    ).strip()


def _endpoint_url() -> str:
    return (getattr(app_config, "HF_QWEN_OCR_ENDPOINT", "") or "").strip().rstrip("/")


def _call_qwen_vision(
    *,
    endpoint: str,
    token: str,
    image_bytes: bytes,
    prompt: str,
    model: str,
    timeout: float,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    t0 = time.perf_counter()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
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
        "max_tokens": 2048,
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    if resp.status_code >= 400:
        text = resp.text[:500]
        if resp.status_code == 503 or "paused" in text.lower():
            raise RuntimeError(
                f"HF Qwen OCR на паузе/не готов (HTTP {resp.status_code}): {text}"
            )
        raise RuntimeError(f"HF Qwen OCR HTTP {resp.status_code}: {text}")
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("HF Qwen OCR: пустой choices")
    message = (choices[0] or {}).get("message") or {}
    raw_text = str(message.get("content") or "").strip()
    parsed = _parse_json_payload(raw_text)
    usage = body.get("usage") or {}
    meta = {
        "provider": "huggingface",
        "endpoint": endpoint,
        "model": body.get("model") or model,
        "tokens_input": usage.get("prompt_tokens"),
        "tokens_output": usage.get("completion_tokens"),
        "raw_preview": raw_text[:400],
    }
    return parsed, wall_ms, meta


def run_qwen_ocr(label_path: Path) -> dict[str, Any]:
    """OCR-only on label crop via HF Qwen2.5-VL endpoint."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "id": "qwen",
        "engine": "qwen",
        "engine_name": "Qwen2.5-VL (HF)",
        "preprocess": "A",
        "preprocess_name": "original",
        "name": "original+Qwen HF",
        "ok": False,
        "prompt": "ocr_only",
    }
    if not getattr(app_config, "USE_QWEN_OCR", True):
        out.update(
            {"skipped": True, "error": "Qwen OCR выключен (USE_QWEN_OCR)"}
        )
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    endpoint = _endpoint_url()
    if not endpoint:
        out.update({"skipped": True, "error": "HF_QWEN_OCR_ENDPOINT не задан"})
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    token = _hf_token()
    if not token:
        out.update({"skipped": True, "error": "HF_ACCESS_TOKEN не задан"})
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    try:
        jpeg, prep_meta = _prepare_jpeg_bytes(label_path)
        out["image_prep"] = prep_meta
        model = (
            getattr(app_config, "HF_QWEN_OCR_MODEL", "") or ""
        ).strip() or "Qwen/Qwen2.5-VL-3B-Instruct"
        out["model"] = model
        timeout = float(
            getattr(app_config, "HF_QWEN_OCR_TIMEOUT_SEC", None)
            or getattr(app_config, "HF_TIMEOUT_SEC", None)
            or 120
        )
        parsed, api_ms, api_meta = _call_qwen_vision(
            endpoint=endpoint,
            token=token,
            image_bytes=jpeg,
            prompt=load_ocr_prompt(),
            model=model,
            timeout=timeout,
        )
        out["api_ms"] = round(api_ms, 1)
        out["model"] = api_meta.get("model") or model
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
            out["error"] = "Qwen OCR вернул пустой full_text"
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
