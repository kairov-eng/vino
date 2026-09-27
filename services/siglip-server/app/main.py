"""SigLIP 2 embedding HTTP service (CPU) + RapidOCR endpoint."""

from __future__ import annotations

import hashlib
import os
import queue
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

import numpy as np
import redis
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = os.environ.get("SIGLIP_MODEL", "google/siglip2-base-patch16-384")
API_KEY = os.environ.get("SIGLIP_API_KEY") or os.environ.get("EMBED_API_KEY", "")
DEVICE = os.environ.get("DEVICE", "cpu")
TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))
REDIS_URL = (os.environ.get("REDIS_URL") or "").strip()
REDIS_TTL = int(os.environ.get("REDIS_CACHE_TTL", "0") or "0")
CACHE_PREFIX = "hash_siglip2_"
# Число движков RapidOCR = сколько OCR-запросов идут параллельно (остальные ждут в очереди).
OCR_WORKERS = max(1, int(os.environ.get("OCR_WORKERS", "2") or "2"))
OCR_WAIT_SEC = float(os.environ.get("OCR_WAIT_SEC", "120") or "120")
# Потоков onnxruntime на движок; -1 (дефолт RapidOCR) = все ядра хоста → движки душат друг друга.
OCR_THREADS = int(os.environ.get("OCR_THREADS", "1") or "1")

# QEMU / x86-64-v1: oneDNN часто падает с "could not create a primitive"
torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.enabled = False

_state: dict[str, Any] = {}
_redis: redis.Redis | None = None


def _check_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _image_hash(raw: bytes) -> str:
    # blake2b быстрее sha256/md5 на типичных размерах JPEG/PNG
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _redis_client() -> redis.Redis | None:
    global _redis
    if not REDIS_URL:
        return None
    if _redis is None:
        _redis = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=False,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
    return _redis


def _cache_get(raw: bytes) -> np.ndarray | None:
    client = _redis_client()
    if client is None:
        return None
    key = CACHE_PREFIX + _image_hash(raw)
    try:
        data = client.get(key)
    except redis.RedisError:
        return None
    if not data:
        return None
    return np.frombuffer(data, dtype=np.float32).copy()


def _cache_set_bytes(key: str, payload: bytes) -> None:
    """Пишется в BackgroundTasks после ответа клиенту."""
    client = _redis_client()
    if client is None:
        return
    try:
        if REDIS_TTL > 0:
            client.set(key, payload, ex=REDIS_TTL)
        else:
            client.set(key, payload)
    except redis.RedisError:
        return


def _load_ocr_pool() -> tuple[queue.Queue, float]:
    from rapidocr_onnxruntime import RapidOCR

    t0 = time.perf_counter()
    pool: queue.Queue = queue.Queue()
    warm = np.full((64, 256, 3), 255, dtype=np.uint8)
    for _ in range(OCR_WORKERS):
        engine = RapidOCR(intra_op_num_threads=OCR_THREADS, inter_op_num_threads=1)
        engine(warm)
        pool.put(engine)
    return pool, round(time.perf_counter() - t0, 3)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)

    try:
        _state["ocr_pool"], _state["ocr_load_sec"] = _load_ocr_pool()
    except Exception as exc:  # noqa: BLE001
        _state["ocr_error"] = str(exc)
        print(f"rapidocr unavailable: {exc}", flush=True)

    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.to(DEVICE)
    model.eval()
    print(f"loaded model class={type(model).__name__}", flush=True)

    # warmup
    dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
    inputs = processor(images=dummy, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        feats = model.get_image_features(**inputs)
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        dim = int(feats.shape[-1])

    redis_ok = False
    if REDIS_URL:
        try:
            redis_ok = bool(_redis_client() and _redis_client().ping())
        except redis.RedisError as exc:
            print(f"redis unavailable: {exc}", flush=True)

    _state["processor"] = processor
    _state["model"] = model
    _state["dim"] = dim
    _state["load_sec"] = round(time.perf_counter() - t0, 3)
    _state["redis_ok"] = redis_ok
    yield
    _state.clear()
    global _redis
    if _redis is not None:
        try:
            _redis.close()
        except Exception:  # noqa: BLE001
            pass
        _redis = None


app = FastAPI(
    title="SigLIP2 Embed",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    root_path="/api_siglip2",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if "model" in _state else "loading",
        "model_id": MODEL_ID,
        "device": DEVICE,
        "dim": _state.get("dim"),
        "load_sec": _state.get("load_sec"),
        "torch_threads": TORCH_THREADS,
        "redis_configured": bool(REDIS_URL),
        "redis_ok": bool(_state.get("redis_ok")),
        "ocr": {
            "engine": "rapidocr_onnxruntime",
            "ready": "ocr_pool" in _state,
            "workers": OCR_WORKERS,
            "threads": OCR_THREADS,
            "load_sec": _state.get("ocr_load_sec"),
            "error": _state.get("ocr_error"),
        },
    }


@app.post("/v1/ocr")
def ocr(
    image: UploadFile = File(...),
    _: None = Depends(_check_key),
) -> dict[str, Any]:
    """RapidOCR по кропу этикетки: строки в порядке движка + score + box."""
    import cv2

    pool: queue.Queue | None = _state.get("ocr_pool")
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail=f"ocr not loaded: {_state.get('ocr_error') or 'loading'}",
        )
    raw = image.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")

    t_total = time.perf_counter()
    t0 = time.perf_counter()
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise HTTPException(status_code=400, detail="cannot decode image")
    decode_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    try:
        engine = pool.get(timeout=OCR_WAIT_SEC)
    except queue.Empty as exc:
        raise HTTPException(status_code=503, detail="ocr busy") from exc
    wait_ms = (time.perf_counter() - t0) * 1000
    try:
        t0 = time.perf_counter()
        result, _elapse = engine(bgr)
        ocr_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"ocr failed: {exc}") from exc
    finally:
        pool.put(engine)

    lines: list[dict[str, Any]] = []
    for item in result or []:
        if not item or len(item) < 2:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        try:
            score = round(float(item[2]), 4)
        except (IndexError, TypeError, ValueError):
            score = None
        try:
            box = [[round(float(x), 1), round(float(y), 1)] for x, y in item[0]]
        except Exception:  # noqa: BLE001
            box = None
        lines.append({"text": text, "score": score, "box": box})

    return {
        "engine": "rapidocr_onnxruntime",
        "text": "\n".join(x["text"] for x in lines),
        "lines": lines,
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "timings_ms": {
            "decode": round(decode_ms, 1),
            "wait": round(wait_ms, 1),
            "ocr": round(ocr_ms, 1),
            "total": round((time.perf_counter() - t_total) * 1000, 1),
        },
        "filename": image.filename,
    }


@app.post("/v1/embed")
async def embed(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    use_cache: str = Form(default="0"),
    _: None = Depends(_check_key),
) -> dict[str, Any]:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")

    want_cache = _truthy(use_cache)
    cache_key = CACHE_PREFIX + _image_hash(raw)
    t_total = time.perf_counter()
    cache_hit = False
    preprocess_ms = 0.0
    infer_ms = 0.0
    cache_ms = 0.0

    if want_cache:
        t0 = time.perf_counter()
        cached = _cache_get(raw)
        cache_ms = (time.perf_counter() - t0) * 1000
        if cached is not None and cached.size > 0:
            vec = cached
            cache_hit = True

    if not cache_hit:
        try:
            t0 = time.perf_counter()
            pil = Image.open(BytesIO(raw)).convert("RGB")
            inputs = _state["processor"](images=pil, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            preprocess_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            with torch.inference_mode():
                feats = _state["model"].get_image_features(**inputs)
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                elif hasattr(feats, "last_hidden_state"):
                    feats = feats.last_hidden_state[:, 0]
                feats = feats.float()
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            infer_ms = (time.perf_counter() - t0) * 1000

            vec = feats.cpu().numpy().astype(np.float32)[0]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"embed failed: {exc}") from exc

        # Всегда кладём в Redis после ответа (не ждём SET на критическом пути)
        payload = np.asarray(vec, dtype=np.float32).tobytes()
        background_tasks.add_task(_cache_set_bytes, cache_key, payload)

    total_ms = (time.perf_counter() - t_total) * 1000
    return {
        "model_id": MODEL_ID,
        "dim": int(vec.shape[0]),
        "embedding": vec.tolist(),
        "l2_norm": float(np.linalg.norm(vec)),
        "timings_ms": {
            "preprocess": round(preprocess_ms, 1),
            "infer": round(infer_ms, 1),
            "cache": round(cache_ms, 1),
            "total": round(total_ms, 1),
        },
        "filename": image.filename,
        "use_cache": want_cache,
        "cache_hit": cache_hit,
        "cache_key": cache_key,
        "cache_write": not cache_hit,
    }
