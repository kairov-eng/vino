"""HTTP clients for SigLIP2 / DINOv3 embeddings.

SigLIP2 cascade (findwine):
  GPU HF → remote CPU (SIGLIP2_ENDPOINT) → in-process local (transformers).

DINOv3: GPU HF → remote CPU (+ optional DINOV3_ENDPOINT_LOCAL).

HF GPU uses a process-wide requests.Session (HTTP keep-alive).
No retries of the same endpoint — one attempt per URL.
"""

from __future__ import annotations

import base64
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from app.db import config as app_config

_hf_session: requests.Session | None = None
_hf_session_lock = threading.Lock()


def get_hf_session() -> requests.Session:
    """Process-wide Session for HF Inference Endpoints (TCP/TLS keep-alive)."""
    global _hf_session
    if _hf_session is not None:
        return _hf_session
    with _hf_session_lock:
        if _hf_session is not None:
            return _hf_session
        sess = requests.Session()
        # No urllib3 retries — findwine owns fallback policy (one attempt / URL).
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _hf_session = sess
        return _hf_session


def warm_hf_connection(endpoint: str, *, timeout: float = 3.0) -> dict[str, Any]:
    """Open keep-alive TCP/TLS to HF host before the real embed POST.

    Uses a cheap GET; 404/405 still establish the connection. Failures are
    non-fatal — the subsequent embed POST will report the real error.
    """
    endpoint = (endpoint or "").strip().rstrip("/")
    out: dict[str, Any] = {"ok": False, "endpoint": endpoint}
    if not endpoint:
        out["error"] = "empty endpoint"
        return out
    sess = get_hf_session()
    t0 = time.perf_counter()
    try:
        resp = sess.get(endpoint, timeout=timeout)
        out["ok"] = True
        out["status_code"] = int(resp.status_code)
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)[:300]
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def _embed_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/v1/embed"):
        return base
    return f"{base}/v1/embed"


def _embedding_device() -> str:
    raw = str(getattr(app_config, "EMBEDDING_DEVICE", "cpu") or "cpu").strip().lower()
    return "gpu" if raw == "gpu" else "cpu"


def call_embed_endpoint(
    image_path: Path,
    *,
    endpoint: str,
    api_key: str | None = None,
    timeout: float | None = None,
    expected_dim: int | None = None,
    use_cache: bool | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """CPU path: own server multipart embed (single attempt)."""
    if api_key is None:
        api_key = app_config.EMBED_API_KEY
    if timeout is None:
        timeout = float(app_config.EMBED_TIMEOUT_SEC)
    if expected_dim is None:
        expected_dim = int(app_config.EMBED_DIM)
    if use_cache is None:
        use_cache = bool(getattr(app_config, "EMBED_CPU_USE_CACHE", False))

    url = _embed_url(endpoint)
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    raw = image_path.read_bytes()
    if not raw:
        raise RuntimeError(f"empty image file: {image_path}")

    form = {"use_cache": "1" if use_cache else "0"}
    files = {
        "image": (
            image_path.name or "image.jpg",
            BytesIO(raw),
            "application/octet-stream",
        )
    }
    t0 = time.perf_counter()
    resp = requests.post(
        url,
        headers=headers,
        files=files,
        data=form,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        raise RuntimeError(f"embed HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    vec = data.get("embedding")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError(f"нет embedding в ответе: {str(data)[:400]}")
    if expected_dim and len(vec) != expected_dim:
        raise RuntimeError(f"ожидался dim={expected_dim}, получили {len(vec)}")
    meta = {
        "provider": "cpu",
        "endpoint": url,
        "dim": int(data.get("dim", len(vec))),
        "model_id": data.get("model_id"),
        "timings_ms": data.get("timings_ms"),
        "l2_norm": data.get("l2_norm"),
        "elapsed_ms": round(elapsed_ms, 1),
        "use_cache": bool(data.get("use_cache", use_cache)),
        "cache_hit": bool(data.get("cache_hit", False)),
    }
    if data.get("cache_key"):
        meta["cache_key"] = data.get("cache_key")
    return [float(x) for x in vec], meta


def call_hf_endpoint(
    image_path: Path,
    *,
    endpoint: str,
    token: str | None = None,
    timeout: float | None = None,
    expected_dim: int | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """GPU path: Hugging Face Inference Endpoint (custom handler, single attempt)."""
    if token is None:
        token = (
            getattr(app_config, "HF_ACCESS_TOKEN", "")
            or getattr(app_config, "HF_TOKEN", "")
            or ""
        ).strip()
    if timeout is None:
        timeout = float(
            getattr(app_config, "HF_TIMEOUT_SEC", None)
            or app_config.EMBED_TIMEOUT_SEC
        )
    if expected_dim is None:
        expected_dim = int(app_config.EMBED_DIM)

    endpoint = endpoint.rstrip("/")
    if not endpoint:
        raise RuntimeError("HF endpoint URL не задан")
    if not token:
        raise RuntimeError("HF_ACCESS_TOKEN не задан")

    raw = image_path.read_bytes()
    if not raw:
        raise RuntimeError(f"empty image file: {image_path}")
    # Prefer JPEG for HF payloads (prepared *_label_hf.jpg is always JPEG)
    suf = image_path.suffix.lower()
    mime = "image/png" if suf == ".png" else "image/jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    payload = {
        "inputs": {"images": [f"data:{mime};base64,{b64}"]},
        "normalize": True,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Connection": "keep-alive",
    }
    t0 = time.perf_counter()
    try:
        resp = get_hf_session().post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"HF endpoint timeout {timeout:g}s (возможно на паузе/не готов): {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"HF endpoint request failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if resp.status_code != 200:
        text = resp.text[:500]
        low = text.lower()
        paused_hint = (
            "paused" in low
            or "на паузе" in low
            or "scale to zero" in low
            or "scaled to zero" in low
            or "maintainer to restart" in low
            or "ask a maintainer" in low
            or "not ready" in low
        )
        if (
            resp.status_code in (503, 502, 529)
            or paused_hint
            or (resp.status_code == 400 and ("pause" in low or "restart" in low))
        ):
            raise RuntimeError(
                f"HF endpoint на паузе/не готов (HTTP {resp.status_code}). "
                f"{text}"
            )
        raise RuntimeError(f"HF HTTP {resp.status_code}: {text}")

    data = resp.json()
    embeddings = data.get("image_embeddings")
    if not embeddings or not isinstance(embeddings, list):
        raise RuntimeError(f"нет image_embeddings в ответе: {str(data)[:400]}")
    vec = embeddings[0]
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("пустой embedding в image_embeddings")
    if expected_dim and len(vec) != expected_dim:
        raise RuntimeError(f"ожидался dim={expected_dim}, получили {len(vec)}")
    meta = {
        "provider": "gpu",
        "endpoint": endpoint,
        "dim": int(data.get("dim", len(vec))),
        "model_id": data.get("model_id"),
        "timings_ms": data.get("timings_ms"),
        "elapsed_ms": round(elapsed_ms, 1),
    }
    return [float(x) for x in vec], meta


def _cpu_endpoints_for(role: str) -> list[str]:
    """Ordered remote CPU bases (HTTP). SigLIP2 local process is separate."""
    role = (role or "").strip().lower()
    if role == "dinov3":
        urls = [
            (getattr(app_config, "DINOV3_ENDPOINT", "") or "").strip().rstrip("/"),
            (getattr(app_config, "DINOV3_ENDPOINT_LOCAL", "") or "").strip().rstrip("/"),
        ]
    else:
        # Remote CPU only — in-process local is a separate stage (not HTTP docker)
        urls = [
            (getattr(app_config, "SIGLIP2_ENDPOINT", "") or "").strip().rstrip("/"),
        ]
    out: list[str] = []
    for u in urls:
        if not u:
            continue
        if u.rstrip("/") not in {x.rstrip("/") for x in out}:
            out.append(u)
    return out


def _call_remote_cpu(
    image_path: Path,
    *,
    role: str,
    use_cache: bool | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Try each remote CPU endpoint once — no same-URL retries."""
    endpoints = _cpu_endpoints_for(role)
    if not endpoints:
        raise RuntimeError(f"CPU endpoint для {role} не задан")
    errors: list[str] = []
    for i, ep in enumerate(endpoints):
        try:
            vec, meta = call_embed_endpoint(
                image_path,
                endpoint=ep,
                use_cache=use_cache,
            )
            if i > 0:
                meta = {
                    **meta,
                    "cpu_endpoint_fallback": True,
                    "cpu_endpoint_tried": endpoints[: i + 1],
                }
            return vec, meta
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{ep}: {exc}")
            continue
    raise RuntimeError(
        "CPU embed failed on all endpoints: " + " | ".join(errors[:4])
    )


def _call_cpu_with_fallback(
    image_path: Path,
    *,
    role: str,
    use_cache: bool | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Remote CPU, then (SigLIP2 only) in-process local."""
    role = (role or "").strip().lower()
    errors: list[str] = []
    try:
        return _call_remote_cpu(image_path, role=role, use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    if role == "siglip2":
        try:
            from app.pipeline.siglip_local import embed_siglip2_local, local_enabled

            if local_enabled():
                vec, meta = embed_siglip2_local(image_path)
                meta = {
                    **meta,
                    "fallback_from_remote_cpu": True,
                    "remote_cpu_error": errors[-1][:400] if errors else None,
                }
                return vec, meta
            errors.append("SIGLIP2_LOCAL_ENABLED=0")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"local: {exc}")
    raise RuntimeError(
        "CPU embed failed (remote + local): " + " | ".join(errors[:4])
    )


def embed_siglip2(
    image_path: Path,
    *,
    force_cpu: bool = False,
    device: str | None = None,
    timeout: float | None = None,
    use_cache: bool | None = None,
) -> tuple[list[float], dict[str, Any]]:
    want = (
        str(device).strip().lower()
        if device is not None
        else _embedding_device()
    )
    if want not in ("cpu", "gpu"):
        want = _embedding_device()
    use_gpu = (not force_cpu) and want == "gpu"
    if use_gpu:
        url = (
            getattr(app_config, "HF_SIGLIP2_ENDPOINT", "")
            or getattr(app_config, "HF_ENDPOINT", "")
            or ""
        ).strip()
        return call_hf_endpoint(image_path, endpoint=url, timeout=timeout)
    vec, meta = _call_cpu_with_fallback(
        image_path, role="siglip2", use_cache=use_cache
    )
    if force_cpu and want == "gpu":
        meta = {**meta, "fallback_from_gpu": True, "requested_device": "gpu"}
    elif want == "gpu" and not use_gpu:
        meta = {**meta, "fallback_from_gpu": True, "requested_device": "gpu"}
    return vec, meta


def embed_siglip2_remote_cpu(
    image_path: Path,
    *,
    use_cache: bool | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Remote SigLIP2 CPU only (no in-process local)."""
    return _call_remote_cpu(image_path, role="siglip2", use_cache=use_cache)


def embed_dinov3(
    image_path: Path,
    *,
    force_cpu: bool = False,
    device: str | None = None,
    timeout: float | None = None,
    use_cache: bool | None = None,
) -> tuple[list[float], dict[str, Any]]:
    want = (
        str(device).strip().lower()
        if device is not None
        else _embedding_device()
    )
    if want not in ("cpu", "gpu"):
        want = _embedding_device()
    use_gpu = (not force_cpu) and want == "gpu"
    if use_gpu:
        url = (getattr(app_config, "HF_DINOV3_ENDPOINT", "") or "").strip()
        return call_hf_endpoint(image_path, endpoint=url, timeout=timeout)
    vec, meta = _call_remote_cpu(image_path, role="dinov3", use_cache=use_cache)
    if force_cpu and want == "gpu":
        meta = {**meta, "fallback_from_gpu": True, "requested_device": "gpu"}
    elif want == "gpu" and not use_gpu:
        meta = {**meta, "fallback_from_gpu": True, "requested_device": "gpu"}
    return vec, meta
