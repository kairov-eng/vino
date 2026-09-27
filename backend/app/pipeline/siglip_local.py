"""In-process SigLIP2 embedding (no Docker / HTTP).

Weights live under backend/models/Siglip2 (SIGLIP2_LOCAL_DIR). On first load,
downloads google/siglip2-base-patch16-384 into that folder if incomplete.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from app.db import config as app_config

_lock = threading.Lock()
_state: dict[str, Any] = {}

_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def _truthy(raw: Any, default: bool = True) -> bool:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if not s:
        return default
    return s in {"1", "true", "yes", "on", "y"}


def local_enabled() -> bool:
    return _truthy(getattr(app_config, "SIGLIP2_LOCAL_ENABLED", True), True)


def _local_dir() -> Path:
    raw = getattr(app_config, "SIGLIP2_LOCAL_DIR", None)
    if raw:
        return Path(raw)
    root = Path(getattr(app_config, "ROOT", Path(__file__).resolve().parents[2]))
    return (root / "models" / "Siglip2").resolve()


def _hub_model_id() -> str:
    return (
        getattr(app_config, "SIGLIP2_LOCAL_MODEL", "")
        or "google/siglip2-base-patch16-384"
    ).strip() or "google/siglip2-base-patch16-384"


def _dir_has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    for pattern in _WEIGHT_GLOBS:
        if any(path.glob(pattern)):
            return True
    # sharded index without loose weight match
    if (path / "model.safetensors.index.json").is_file():
        return True
    return False


def _hub_cache_snapshot(hub_id: str) -> Path | None:
    """Return path to a complete HF hub cache snapshot, if any."""
    home = Path.home() / ".cache" / "huggingface" / "hub"
    # google/siglip2-base-patch16-384 → models--google--siglip2-base-patch16-384
    folder = "models--" + hub_id.replace("/", "--")
    root = home / folder
    if not root.is_dir():
        return None
    refs_main = root / "refs" / "main"
    rev = None
    if refs_main.is_file():
        rev = refs_main.read_text(encoding="utf-8").strip() or None
    snaps = root / "snapshots"
    if not snaps.is_dir():
        return None
    if rev and (snaps / rev).is_dir():
        cand = snaps / rev
        if _dir_has_weights(cand):
            return cand
    # newest snapshot with weights
    for cand in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if cand.is_dir() and _dir_has_weights(cand):
            return cand
    return None


def _copy_snapshot_to_dir(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name.startswith("."):
            continue
        target = dest / item.name
        # HF cache may use symlinks/hardlinks — copy real bytes
        real = item.resolve() if item.exists() else item
        if real.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(real, target)
        else:
            shutil.copy2(real, target)


def ensure_model_dir(*, download: bool = True) -> Path:
    """Return SIGLIP2_LOCAL_DIR; fill from HF cache or hub download if needed."""
    dest = _local_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if _dir_has_weights(dest):
        return dest
    if not download:
        raise RuntimeError(
            f"SigLIP2 weights missing in {dest} "
            f"(need config.json + *.safetensors|*.bin)"
        )
    hub_id = _hub_model_id()

    # Prefer local HF cache (avoids re-download / SSL flakiness)
    cached = _hub_cache_snapshot(hub_id)
    if cached is not None:
        _copy_snapshot_to_dir(cached, dest)
        if _dir_has_weights(dest):
            return dest

    token = (
        getattr(app_config, "HF_ACCESS_TOKEN", "")
        or getattr(app_config, "HF_TOKEN", "")
        or None
    )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "нужен huggingface_hub для скачивания SigLIP2 "
            "(pip install huggingface_hub)"
        ) from exc
    # .env HF_ENDPOINT = Inference Endpoint URL; hub download must use huggingface.co
    old_hf_endpoint = os.environ.pop("HF_ENDPOINT", None)
    try:
        snapshot_download(
            repo_id=hub_id,
            local_dir=str(dest),
            token=token or None,
            endpoint="https://huggingface.co",
            max_workers=2,
        )
    except Exception as exc:  # noqa: BLE001
        cached = _hub_cache_snapshot(hub_id)
        if cached is not None:
            _copy_snapshot_to_dir(cached, dest)
            if _dir_has_weights(dest):
                return dest
        raise RuntimeError(
            f"не удалось скачать {hub_id} в {dest}: {exc}"
        ) from exc
    finally:
        if old_hf_endpoint is not None:
            os.environ["HF_ENDPOINT"] = old_hf_endpoint
    if not _dir_has_weights(dest):
        raise RuntimeError(
            f"после download нет весов в {dest} (hub={hub_id})"
        )
    return dest


def _resolve_device(want: str) -> str:
    want = (want or "cpu").strip().lower()
    if want == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    if want in ("cuda", "gpu"):
        return "cuda"
    return "cpu"


def _ensure_loaded() -> dict[str, Any]:
    if "model" in _state:
        return _state
    with _lock:
        if "model" in _state:
            return _state
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "локальный SigLIP2 требует torch и transformers "
                "(pip install torch transformers)"
            ) from exc

        model_path = ensure_model_dir(download=True)
        model_id = str(model_path)
        device = _resolve_device(
            getattr(app_config, "SIGLIP2_LOCAL_DEVICE", "cpu") or "cpu"
        )
        threads = int(
            getattr(app_config, "SIGLIP2_LOCAL_TORCH_THREADS", 2) or 2
        )
        if threads > 0:
            try:
                torch.set_num_threads(threads)
                torch.set_num_interop_threads(1)
            except Exception:  # noqa: BLE001
                pass
        # Avoid oneDNN crashes on some CPUs (same as siglip-server)
        try:
            torch.backends.mkldnn.enabled = False
        except Exception:  # noqa: BLE001
            pass

        t0 = time.perf_counter()
        processor = AutoImageProcessor.from_pretrained(
            model_id, use_fast=False, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float32, local_files_only=True
        )
        model.to(device)
        model.eval()

        # Warmup
        from PIL import Image

        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        inputs = processor(images=dummy, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            dim = int(feats.shape[-1])

        _state["processor"] = processor
        _state["model"] = model
        _state["device"] = device
        _state["model_id"] = _hub_model_id()
        _state["model_dir"] = model_id
        _state["dim"] = dim
        _state["load_sec"] = round(time.perf_counter() - t0, 3)
        return _state


def embed_siglip2_local(image_path: Path) -> tuple[list[float], dict[str, Any]]:
    """Compute SigLIP2 image embedding in-process (CPU/CUDA)."""
    if not local_enabled():
        raise RuntimeError("SIGLIP2_LOCAL_ENABLED=0")

    import torch
    from PIL import Image

    st = _ensure_loaded()
    processor = st["processor"]
    model = st["model"]
    device = st["device"]
    expected_dim = int(getattr(app_config, "EMBED_DIM", 768) or 768)

    raw = Path(image_path).read_bytes()
    if not raw:
        raise RuntimeError(f"empty image file: {image_path}")

    t0 = time.perf_counter()
    pil = Image.open(image_path).convert("RGB")
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    preprocess_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    with torch.inference_mode():
        feats = model.get_image_features(**inputs)
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    infer_ms = (time.perf_counter() - t1) * 1000

    vec = feats.detach().cpu().numpy().astype("float32")[0]
    if expected_dim and len(vec) != expected_dim:
        raise RuntimeError(
            f"ожидался dim={expected_dim}, получили {len(vec)}"
        )
    elapsed_ms = preprocess_ms + infer_ms
    meta: dict[str, Any] = {
        "provider": "local",
        "device": "local",
        "endpoint": "local",
        "dim": int(len(vec)),
        "model_id": st.get("model_id"),
        "model_dir": st.get("model_dir"),
        "torch_device": device,
        "load_sec": st.get("load_sec"),
        "timings_ms": {
            "preprocess": round(preprocess_ms, 1),
            "infer": round(infer_ms, 1),
            "total": round(elapsed_ms, 1),
        },
        "elapsed_ms": round(elapsed_ms, 1),
        "l2_norm": float((vec ** 2).sum() ** 0.5),
    }
    return [float(x) for x in vec], meta
