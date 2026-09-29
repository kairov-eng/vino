# =============================================================================
# gemini_proxy — HTTP-прокси к Google Gemini API
# =============================================================================
#
# Endpoint'ы:
#   GET  /health
#   POST /v1/vision/json  — картинка + prompt → JSON от Gemini
#     multipart/form-data:
#       file   — изображение (обязательно)
#       prompt — текст промпта (обязательно)
#       model  — опционально (иначе GEMINI_MODEL из env)
#
# Ключи GEMINI_API_KEY_1..N и лимит 15 req/min на ключ — внутри прокси.
# Клиент (crop_heic_folder_server) ходит только сюда, без прямого вызова Google.
#
# =============================================================================

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

MAX_RPM_PER_KEY = int(os.getenv("GEMINI_MAX_RPM", "15"))
RPM_WINDOW_SEC = float(os.getenv("GEMINI_RPM_WINDOW_SEC", "60"))
DEFAULT_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-3.6-flash").strip()


def load_api_keys() -> list[str]:
    keys: list[str] = []
    i = 1
    while True:
        val = (os.getenv(f"GEMINI_API_KEY_{i}") or "").strip()
        if not val:
            break
        keys.append(val)
        i += 1
    legacy = (os.getenv("GEMINI_API_KEY") or "").strip()
    if legacy and legacy not in keys:
        keys.append(legacy)
    return keys


class KeyRateLimiter:
    def __init__(self, label: str, max_rpm: int = MAX_RPM_PER_KEY, window: float = RPM_WINDOW_SEC):
        self.label = label
        self.max_rpm = max_rpm
        self.window = window
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self.window:
                    self._times.popleft()
                if len(self._times) < self.max_rpm:
                    self._times.append(now)
                    return
                wait = self.window - (now - self._times[0]) + 0.01
            time.sleep(max(wait, 0.05))


class KeyPool:
    """Round-robin по ключам с per-key rate limit."""

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise RuntimeError("Нет GEMINI_API_KEY_*")
        self.keys = keys
        self.limiters = [
            KeyRateLimiter(f"key{i + 1}/...{k[-6:]}") for i, k in enumerate(keys)
        ]
        self._idx = 0
        self._lock = threading.Lock()

    def next(self) -> tuple[str, KeyRateLimiter]:
        with self._lock:
            i = self._idx % len(self.keys)
            self._idx += 1
            return self.keys[i], self.limiters[i]


API_KEYS = load_api_keys()
POOL = KeyPool(API_KEYS) if API_KEYS else None

app = FastAPI(title="Gemini Vision Proxy", version="1.0.0")


class HealthOut(BaseModel):
    ok: bool
    model: str
    keys: int
    max_rpm_per_key: int


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        ok=POOL is not None,
        model=DEFAULT_MODEL,
        keys=len(API_KEYS),
        max_rpm_per_key=MAX_RPM_PER_KEY,
    )


def _extract_json(text: str) -> dict[str, Any]:
    import json
    import re

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


def _usage_tokens(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    tin = getattr(usage, "prompt_token_count", None) or getattr(usage, "promptTokenCount", None)
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


def call_gemini(image_bytes: bytes, mime: str, prompt: str, model: str) -> dict[str, Any]:
    if POOL is None:
        raise HTTPException(status_code=503, detail="Нет GEMINI_API_KEY_* в окружении прокси")

    # Неизвестные/устаревшие id (напр. gemini-3.5-flash-lite) → 400 INVALID_ARGUMENT.
    # Пробуем запрошенную модель, затем DEFAULT_MODEL.
    models: list[str] = []
    for m in (model, DEFAULT_MODEL):
        mid = (m or "").strip()
        if mid and mid not in models:
            models.append(mid)
    if not models:
        raise HTTPException(status_code=500, detail="GEMINI_MODEL не задан")

    last_error: Exception | None = None
    for mid in models:
        # По каждому ключу один раз на эту модель
        for _ in range(len(API_KEYS)):
            api_key, limiter = POOL.next()
            limiter.acquire()
            t0 = time.perf_counter()
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=mid,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                elapsed = time.perf_counter() - t0
                tokens_in, tokens_out = _usage_tokens(response)
                raw = (response.text or "").strip()
                if not raw:
                    raise RuntimeError("Пустой ответ модели")
                data = _extract_json(raw)
                return {
                    "ok": True,
                    "data": data,
                    "elapsed_sec": round(elapsed, 3),
                    "tokens_input": tokens_in,
                    "tokens_output": tokens_out,
                    "model": mid,
                    "requested_model": model,
                    "key_label": limiter.label,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                # 400/404 по модели — сразу следующая модель, не крутим все ключи зря
                err_s = str(exc)
                if "INVALID_ARGUMENT" in err_s or "NOT_FOUND" in err_s or "404" in err_s:
                    break
                continue

    raise HTTPException(
        status_code=502,
        detail=f"Все ключи Gemini не сработали (models={models}): {last_error}",
    )


@app.post("/v1/vision/json")
async def vision_json(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    model: str | None = Form(None),
) -> dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл")
    mime = file.content_type or "image/jpeg"
    if mime == "application/octet-stream":
        mime = "image/jpeg"
    mid = (model or DEFAULT_MODEL).strip()
    return call_gemini(image_bytes, mime, prompt, mid)
