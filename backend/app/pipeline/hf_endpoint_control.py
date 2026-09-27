"""Wake / resume Hugging Face Inference Endpoints used for GPU embeddings."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from app.db import config as app_config


def _token() -> str:
    return (
        getattr(app_config, "HF_ACCESS_TOKEN", "")
        or getattr(app_config, "HF_TOKEN", "")
        or ""
    ).strip()


def _namespace() -> str:
    ns = (getattr(app_config, "HF_NAMESPACE", "") or "").strip()
    if ns:
        return ns
    try:
        from huggingface_hub import whoami

        info = whoami(token=_token())
        return str(info.get("name") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _norm_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    try:
        host = urlparse(u).netloc.lower()
        return host or u.lower()
    except Exception:  # noqa: BLE001
        return u.lower()


def _endpoint_url_for_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role == "siglip2":
        return (
            getattr(app_config, "HF_SIGLIP2_ENDPOINT", "")
            or getattr(app_config, "HF_ENDPOINT", "")
            or ""
        ).strip().rstrip("/")
    if role == "dinov3":
        return (getattr(app_config, "HF_DINOV3_ENDPOINT", "") or "").strip().rstrip("/")
    if role == "qwen_ocr":
        return (
            getattr(app_config, "HF_QWEN_OCR_ENDPOINT", "") or ""
        ).strip().rstrip("/")
    return ""


def _explicit_name_for_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role == "siglip2":
        return (getattr(app_config, "HF_SIGLIP2_ENDPOINT_NAME", "") or "").strip()
    if role == "dinov3":
        return (getattr(app_config, "HF_DINOV3_ENDPOINT_NAME", "") or "").strip()
    if role == "qwen_ocr":
        return (getattr(app_config, "HF_QWEN_OCR_ENDPOINT_NAME", "") or "").strip()
    return ""


def _wanted_endpoint_urls(
    *,
    use_siglip2: bool = True,
    use_dinov3: bool = False,
    use_qwen_ocr: bool = False,
) -> list[tuple[str, str]]:
    """Return [(role, url), ...] for endpoints needed this run."""
    out: list[tuple[str, str]] = []
    if use_siglip2:
        siglip = _endpoint_url_for_role("siglip2")
        if siglip:
            out.append(("siglip2", siglip))
    if use_dinov3:
        dino = _endpoint_url_for_role("dinov3")
        if dino:
            out.append(("dinov3", dino))
    if use_qwen_ocr:
        qwen = _endpoint_url_for_role("qwen_ocr")
        if qwen:
            out.append(("qwen_ocr", qwen))
    return out


def _resolve_name_for_url(
    url: str,
    *,
    namespace: str,
    token: str,
    explicit_name: str | None = None,
) -> str | None:
    if explicit_name:
        return explicit_name.strip() or None
    host = _norm_url(url)
    if not host:
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for ep in api.list_inference_endpoints(namespace=namespace):
        ep_url = getattr(ep, "url", None) or ""
        if _norm_url(str(ep_url)) == host:
            return str(getattr(ep, "name", "") or "") or None
    return None


def _ensure_one_running(
    *,
    name: str,
    namespace: str,
    token: str,
    role: str,
    url: str,
    timeout_sec: float,
    wait: bool = True,
) -> dict[str, Any]:
    from huggingface_hub import get_inference_endpoint

    t0 = time.perf_counter()
    item: dict[str, Any] = {
        "role": role,
        "name": name,
        "namespace": namespace,
        "url": url,
        "ok": False,
        "provider": "huggingface",
        "action": "start",
        "waited": bool(wait),
    }
    try:
        ep = get_inference_endpoint(name, namespace=namespace, token=token)
        status0 = str(getattr(ep, "status", "") or "")
        item["status_before"] = status0
        running_like = {"running", "ready"}
        if status0.lower() not in running_like:
            ep.resume(running_ok=True)
            item["resumed"] = True
        else:
            item["resumed"] = False
        if not wait:
            # Fire-and-forget: resume issued, do not block current scan
            try:
                status1 = str(getattr(ep, "status", "") or status0)
            except Exception:  # noqa: BLE001
                status1 = status0
            item["status_after"] = status1
            item["ok"] = True
            item["note"] = "resume без ожидания готовности"
            item["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return item
        wait_timeout = int(max(1, timeout_sec))
        ep.wait(timeout=wait_timeout, refresh_every=5)
        status1 = str(getattr(ep, "status", "") or "")
        item["status_after"] = status1
        item["url"] = getattr(ep, "url", None) or url
        item["ok"] = status1.lower() in running_like
        if not item["ok"]:
            item["error"] = f"endpoint status after wait: {status1}"
    except Exception as exc:  # noqa: BLE001
        item["error"] = str(exc)
    item["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return item


def is_hf_endpoint_paused_error(exc: BaseException | str) -> bool:
    """True if HF Inference Endpoint is paused / scaled to zero / not ready."""
    text = str(exc or "").lower()
    needles = (
        "paused",
        "на паузе",
        "503",
        "service unavailable",
        "scale to zero",
        "scaled to zero",
        "not ready",
        "stopped",
        "offline",
        "initializing",
        "pending",
        "deploying",
        "endpoint is not running",
        "no instance",
        "timeout",
        "timed out",
        "maintainer to restart",
        "ask a maintainer",
        "bad_request",
    )
    if any(n in text for n in needles):
        return True
    # HTTP 400 + pause wording (HF Inference returns BAD_REQUEST when paused)
    if "http 400" in text and (
        "pause" in text or "restart" in text or "not ready" in text
    ):
        return True
    return False


def is_hf_endpoint_network_error(exc: BaseException | str) -> bool:
    """True for SSL / connection / EOF — kick resume for next scan."""
    text = str(exc or "").lower()
    needles = (
        "ssl",
        "eof",
        "connection",
        "max retries",
        "reset by peer",
        "broken pipe",
        "name resolution",
        "getaddrinfo",
        "unexpected_eof",
    )
    return any(n in text for n in needles)


def start_hf_endpoint(
    role: str,
    *,
    timeout_sec: float | None = None,
    wait: bool = True,
) -> dict[str, Any]:
    """Resume HF endpoint (siglip2 | dinov3 | qwen_ocr).

    wait=True  — resume + ep.wait (blocking until ready or timeout).
    wait=False — только resume, сразу return (для fallback-скана в фоне).
    """
    t0 = time.perf_counter()
    role = (role or "").strip().lower()
    out: dict[str, Any] = {
        "ok": False,
        "role": role,
        "provider": "huggingface",
        "action": "start",
        "waited": bool(wait),
    }
    if role not in ("siglip2", "dinov3", "qwen_ocr"):
        out["error"] = f"unknown role: {role}"
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    token = _token()
    if not token:
        out["error"] = "HF_ACCESS_TOKEN не задан"
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    namespace = _namespace()
    if not namespace:
        out["error"] = "HF_NAMESPACE не задан и whoami не вернул name"
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    out["namespace"] = namespace

    url = _endpoint_url_for_role(role)
    if not url:
        out["error"] = f"нет HF URL для {role}"
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    out["url"] = url

    if timeout_sec is None:
        timeout_sec = float(
            getattr(app_config, "HF_START_TIMEOUT_SEC", None) or 15
        )
    timeout_sec = max(1.0, float(timeout_sec))
    out["timeout_sec"] = timeout_sec

    try:
        name = _resolve_name_for_url(
            url,
            namespace=namespace,
            token=token,
            explicit_name=_explicit_name_for_role(role) or None,
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"resolve name: {exc}"
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    if not name:
        out["error"] = "не найден Inference Endpoint с таким URL"
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    item = _ensure_one_running(
        name=name,
        namespace=namespace,
        token=token,
        role=role,
        url=url,
        timeout_sec=timeout_sec,
        wait=wait,
    )
    # Prefer wall time from ensure; keep role-level keys
    out.update(item)
    if "ms" not in out or not out["ms"]:
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def start_pipeline_hf_endpoints(
    *,
    use_siglip2: bool = True,
    use_dinov3: bool = False,
    use_qwen_ocr: bool = False,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Resume + wait HF endpoints needed for GPU embed / Qwen OCR (parallel)."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "ok": False,
        "provider": "huggingface",
        "action": "start",
        "endpoints": [],
    }
    wanted = _wanted_endpoint_urls(
        use_siglip2=use_siglip2,
        use_dinov3=use_dinov3,
        use_qwen_ocr=use_qwen_ocr,
    )
    if not wanted:
        out["error"] = "нет HF_*_ENDPOINT URL в конфиге"
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    if timeout_sec is None:
        timeout_sec = float(
            getattr(app_config, "HF_START_TIMEOUT_SEC", None) or 15
        )
    timeout_sec = max(1.0, float(timeout_sec))
    out["timeout_sec"] = timeout_sec

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, len(wanted))) as pool:
        futs = {
            pool.submit(start_hf_endpoint, role, timeout_sec=timeout_sec): role
            for role, _url in wanted
        }
        for fut in as_completed(futs):
            out["endpoints"].append(fut.result())

    out["ok"] = bool(out["endpoints"]) and all(
        bool(x.get("ok")) for x in out["endpoints"]
    )
    if not out["ok"] and not out.get("error"):
        errs = [x.get("error") for x in out["endpoints"] if x.get("error")]
        out["error"] = "; ".join(str(e) for e in errs if e) or "HF start failed"
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
