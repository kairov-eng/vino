"""Gemini vision OCR for a single label image.

Transport (priority):
  1) GEMINI_FORCE_PROXY=1 → HTTP gemini-proxy
  2) GOOGLE_AI_API_VERTEX=1 → Vertex AI (service account), как label_detect/gemini_client.py
  3) GEMINI_API_KEY[_N] → Google AI Studio (direct)
  4) GEMINI_PROXY_URL → HTTP proxy fallback
"""

from __future__ import annotations

import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from app.db import config as app_config
from app.pipeline.gemini_client import (
    describe_backend,
    make_genai_client,
    use_vertex_api,
)
from app.pipeline.label_detect_llm import (
    finalize_detect_result,
    load_label_ocr_prompt,
    load_ocr_prompt,
    resolve_image_sizes,
)
from app.pipeline.ocr_match import normalize_ocr_text


def _prepare_jpeg_bytes(image_path: Path) -> tuple[bytes, dict[str, Any]]:
    """Downscale long side and compress JPEG for upload."""
    meta: dict[str, Any] = {
        "source_path": str(image_path),
        "max_side": app_config.GEMINI_OCR_MAX_SIDE,
        "max_bytes": app_config.GEMINI_OCR_MAX_BYTES,
    }
    img = Image.open(image_path)
    img = img.convert("RGB")
    w0, h0 = img.size
    meta["original_size"] = [w0, h0]

    max_side = max(1, int(app_config.GEMINI_OCR_MAX_SIDE))
    scale = min(1.0, max_side / float(max(w0, h0)))
    if scale < 1.0:
        nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        meta["resized"] = True
        meta["resized_size"] = [nw, nh]
    else:
        meta["resized"] = False
        meta["resized_size"] = [w0, h0]

    quality = 85
    data = b""
    for q in (quality, 75, 65, 55, 45):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) <= int(app_config.GEMINI_OCR_MAX_BYTES):
            quality = q
            break
    while len(data) > int(app_config.GEMINI_OCR_MAX_BYTES) and max(img.size) > 512:
        nw = max(1, int(img.size[0] * 0.85))
        nh = max(1, int(img.size[1] * 0.85))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=max(40, quality - 10), optimize=True)
        data = buf.getvalue()
        meta["resized"] = True
        meta["resized_size"] = [nw, nh]

    meta["jpeg_bytes"] = len(data)
    meta["jpeg_quality"] = quality
    return data, meta


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def _usage_tokens(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    tin = getattr(usage, "prompt_token_count", None) or getattr(
        usage, "promptTokenCount", None
    )
    tout = getattr(usage, "candidates_token_count", None) or getattr(
        usage, "candidatesTokenCount", None
    )
    try:
        tin_i = int(tin) if tin is not None else None
    except (TypeError, ValueError):
        tin_i = None
    try:
        tout_i = int(tout) if tout is not None else None
    except (TypeError, ValueError):
        tout_i = None
    return tin_i, tout_i


def _use_vertex_gemini() -> bool:
    if bool(getattr(app_config, "GEMINI_FORCE_PROXY", False)):
        return False
    return use_vertex_api()


def _use_direct_gemini() -> bool:
    """AI Studio API keys (не Vertex)."""
    if bool(getattr(app_config, "GEMINI_FORCE_PROXY", False)):
        return False
    if use_vertex_api():
        return False
    keys = getattr(app_config, "GEMINI_API_KEYS", None) or []
    return bool(keys)


def _gemini_transport_ready() -> bool:
    if bool(getattr(app_config, "GEMINI_FORCE_PROXY", False)):
        return bool((app_config.GEMINI_PROXY_URL or "").strip())
    if use_vertex_api():
        return True
    if getattr(app_config, "GEMINI_API_KEYS", None):
        return True
    return bool((app_config.GEMINI_PROXY_URL or "").strip())


def _call_gemini_vertex(
    *,
    image_bytes: bytes,
    prompt: str,
    model: str,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Vertex AI via google-genai (vertexai=True + SA credentials)."""
    from google.genai import types

    configs = [
        types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        types.GenerateContentConfig(temperature=0.1),
    ]
    models: list[str] = []
    for mid in (
        model,
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash",
        "gemini-flash-latest",
    ):
        m = (mid or "").strip()
        if m and m not in models:
            models.append(m)

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        prompt,
    ]
    client = make_genai_client(model=model)
    backend = describe_backend(model=model)
    last_exc: Exception | None = None
    attempts: list[str] = []
    for mid in models:
        for ci, cfg in enumerate(configs):
            t0 = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model=mid,
                    contents=contents,
                    config=cfg,
                )
                api_ms = (time.perf_counter() - t0) * 1000
                raw = (getattr(response, "text", None) or "").strip()
                if not raw:
                    raise RuntimeError("пустой ответ модели")
                parsed = _extract_json(raw)
                tokens_in, tokens_out = _usage_tokens(response)
                meta = {
                    "transport": "vertex",
                    "backend": backend,
                    "tokens_input": tokens_in,
                    "tokens_output": tokens_out,
                    "proxy_model": mid,
                    "requested_model": model,
                    "config_idx": ci,
                    "wall_ms": round(api_ms, 1),
                }
                return parsed, api_ms, meta
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                attempts.append(f"{mid}/cfg{ci}: {exc}")
                err_s = str(exc)
                if (
                    "NOT_FOUND" in err_s
                    or "404" in err_s
                    or "no longer available" in err_s
                ):
                    break
    assert last_exc is not None
    detail = attempts[-1] if attempts else str(last_exc)
    raise RuntimeError(
        f"Gemini Vertex: модели не сработали (tried={models}): {detail}"
    ) from last_exc


def _call_gemini_direct(
    *,
    image_bytes: bytes,
    prompt: str,
    model: str,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Direct Google AI Studio — api_key."""
    from google.genai import types

    keys: list[str] = list(getattr(app_config, "GEMINI_API_KEYS", None) or [])
    if not keys:
        raise RuntimeError("Нет GEMINI_API_KEY / GEMINI_API_KEY_*")

    configs = [
        types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        types.GenerateContentConfig(temperature=0.1),
    ]
    models: list[str] = []
    for mid in (
        model,
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash",
        "gemini-flash-latest",
    ):
        m = (mid or "").strip()
        if m and m not in models:
            models.append(m)

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        prompt,
    ]
    last_exc: Exception | None = None
    attempts: list[str] = []
    for mid in models:
        model_gone = False
        for ki, api_key in enumerate(keys):
            client = make_genai_client(api_key=api_key, model=mid)
            for ci, cfg in enumerate(configs):
                t0 = time.perf_counter()
                try:
                    response = client.models.generate_content(
                        model=mid,
                        contents=contents,
                        config=cfg,
                    )
                    api_ms = (time.perf_counter() - t0) * 1000
                    raw = (getattr(response, "text", None) or "").strip()
                    if not raw:
                        raise RuntimeError("пустой ответ модели")
                    parsed = _extract_json(raw)
                    tokens_in, tokens_out = _usage_tokens(response)
                    meta = {
                        "transport": "direct",
                        "backend": describe_backend(api_key=api_key, model=mid),
                        "key_label": f"key{ki + 1}/...{api_key[-6:]}",
                        "tokens_input": tokens_in,
                        "tokens_output": tokens_out,
                        "proxy_model": mid,
                        "requested_model": model,
                        "config_idx": ci,
                        "wall_ms": round(api_ms, 1),
                    }
                    return parsed, api_ms, meta
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    attempts.append(f"{mid}/cfg{ci}/key{ki + 1}: {exc}")
                    err_s = str(exc)
                    if (
                        "NOT_FOUND" in err_s
                        or "404" in err_s
                        or "no longer available" in err_s
                    ):
                        model_gone = True
                        break
            if model_gone:
                break
    assert last_exc is not None
    detail = attempts[-1] if attempts else str(last_exc)
    raise RuntimeError(
        f"Gemini direct: все ключи/модели не сработали "
        f"(tried={models}): {detail}"
    ) from last_exc


def _call_gemini_proxy(
    *,
    proxy_url: str,
    image_bytes: bytes,
    prompt: str,
    model: str,
    timeout: float,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """POST /v1/vision/json — как в 3. crop_heic_folder_server.py."""
    t0 = time.perf_counter()
    files = {"file": ("label.jpg", image_bytes, "image/jpeg")}
    form = {"prompt": prompt, "model": model}
    resp = requests.post(
        f"{proxy_url}/v1/vision/json",
        files=files,
        data=form,
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    if resp.status_code >= 400:
        detail = resp.text[:500]
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"gemini-proxy HTTP {resp.status_code}: {detail}")
    body = resp.json()
    if not body.get("ok"):
        raise RuntimeError(f"gemini-proxy error: {body}")
    parsed = body.get("data")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"gemini-proxy: data не объект: {type(parsed)}")
    elapsed_sec = body.get("elapsed_sec")
    api_ms = float(elapsed_sec) * 1000 if elapsed_sec is not None else wall_ms
    meta = {
        "transport": "proxy",
        "proxy_url": proxy_url,
        "key_label": body.get("key_label"),
        "tokens_input": body.get("tokens_input"),
        "tokens_output": body.get("tokens_output"),
        "proxy_model": body.get("model"),
        "wall_ms": round(wall_ms, 1),
    }
    return parsed, api_ms, meta


def _call_gemini(
    *,
    image_bytes: bytes,
    prompt: str,
    model: str,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    if bool(getattr(app_config, "GEMINI_FORCE_PROXY", False)):
        proxy_url = (app_config.GEMINI_PROXY_URL or "").strip().rstrip("/")
        if not proxy_url:
            raise RuntimeError("GEMINI_FORCE_PROXY=1, но GEMINI_PROXY_URL пуст")
        return _call_gemini_proxy(
            proxy_url=proxy_url,
            image_bytes=image_bytes,
            prompt=prompt,
            model=model,
            timeout=float(app_config.GEMINI_OCR_TIMEOUT_SEC),
        )
    if _use_vertex_gemini():
        return _call_gemini_vertex(
            image_bytes=image_bytes, prompt=prompt, model=model
        )
    if _use_direct_gemini():
        return _call_gemini_direct(
            image_bytes=image_bytes, prompt=prompt, model=model
        )
    proxy_url = (app_config.GEMINI_PROXY_URL or "").strip().rstrip("/")
    if not proxy_url:
        raise RuntimeError(
            "Нет Vertex/GEMINI_API_KEY_* и не задан GEMINI_PROXY_URL"
        )
    return _call_gemini_proxy(
        proxy_url=proxy_url,
        image_bytes=image_bytes,
        prompt=prompt,
        model=model,
        timeout=float(app_config.GEMINI_OCR_TIMEOUT_SEC),
    )


def run_gemini_ocr(label_path: Path) -> dict[str, Any]:
    """OCR label via Gemini; returns status-compatible dict with JSON fields."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "id": "gemini",
        "engine": "gemini",
        "engine_name": "Google Gemini",
        "preprocess": "A",
        "preprocess_name": "original",
        "name": "original+Gemini",
        "ok": False,
    }
    if not app_config.USE_GEMINI_OCR:
        out.update({"skipped": True, "error": "Gemini OCR выключен (USE_GEMINI_OCR)"})
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    if not _gemini_transport_ready():
        out.update(
            {
                "skipped": True,
                "error": (
                    "Нет Vertex (GOOGLE_AI_API_VERTEX) / GEMINI_API_KEY_* "
                    "и не задан GEMINI_PROXY_URL"
                ),
            }
        )
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    try:
        jpeg, prep_meta = _prepare_jpeg_bytes(label_path)
        out["image_prep"] = prep_meta
        model = (app_config.GEMINI_MODEL or "").strip() or "gemini-3.6-flash"
        out["model"] = model
        parsed, api_ms, call_meta = _call_gemini(
            image_bytes=jpeg,
            prompt=load_ocr_prompt(),
            model=model,
        )
        out["api_ms"] = round(api_ms, 1)
        out["model"] = call_meta.get("proxy_model") or model
        out["proxy"] = call_meta
        tokens_in = call_meta.get("tokens_input")
        tokens_out = call_meta.get("tokens_output")
        out["tokens_input"] = tokens_in
        out["tokens_output"] = tokens_out

        full = str(parsed.get("full_text") or "").strip()
        if not full and isinstance(parsed.get("lines"), list):
            full = "\n".join(str(x).strip() for x in parsed["lines"] if str(x).strip())
        lines = []
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
            out["error"] = "Gemini вернул пустой full_text"
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def run_gemini_label_detect_ocr(
    image_path: Path,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Detect label boxes + OCR on full frame (same prompt as OpenAI label+OCR)."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "id": "gemini_label_detect",
        "engine": "gemini",
        "engine_name": "Gemini label+OCR",
        "ok": False,
        "prompt": "label_ocr",
        "boxes": [],
        "selected": None,
    }
    if not _gemini_transport_ready():
        out.update(
            {
                "skipped": True,
                "error": (
                    "Нет Vertex (GOOGLE_AI_API_VERTEX) / GEMINI_API_KEY_* "
                    "и не задан GEMINI_PROXY_URL"
                ),
            }
        )
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    try:
        jpeg, prep_meta = _prepare_jpeg_bytes(image_path)
        out["image_prep"] = prep_meta
        sent_size, orig_size = resolve_image_sizes(image_path, prep_meta, image_size)

        model = (app_config.GEMINI_MODEL or "").strip() or "gemini-3.6-flash"
        out["model"] = model
        parsed, api_ms, call_meta = _call_gemini(
            image_bytes=jpeg,
            prompt=load_label_ocr_prompt(),
            model=model,
        )
        out.update(
            finalize_detect_result(
                engine="gemini",
                engine_name="Gemini",
                parsed=parsed,
                api_ms=api_ms,
                tokens_in=call_meta.get("tokens_input"),
                tokens_out=call_meta.get("tokens_output"),
                model=call_meta.get("proxy_model") or model,
                api_meta=call_meta,
                sent_size=sent_size,
                orig_size=orig_size,
            )
        )
        out["proxy"] = call_meta
        out["model"] = out.get("model") or model
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
