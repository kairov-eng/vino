"""
Custom Inference Endpoint handler for DINOv3 image embeddings.

Uses facebook/dinov3-vitb16-pretrain-lvd1689m (768-d via pooler_output + L2).

Request examples:

  {"inputs": {"images": ["data:image/jpeg;base64,..."]}}
  {"inputs": {"images": ["https://.../photo.jpg"]}}
  {"inputs": ["data:image/png;base64,..."], "normalize": true}

Response:

  {
    "model_id": "...",
    "dim": 768,
    "image_embeddings": [[...], ...],
    "timings_ms": {"preprocess": ..., "infer": ..., "total": ...}
  }
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _hub_token() -> str | None:
    """Токен для gated facebook/dinov3 (HF Endpoint / UI Secrets)."""
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_API_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    ):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _load_image(value: str | bytes | Image.Image) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")

    if not isinstance(value, str):
        raise ValueError(f"unsupported image input type: {type(value)}")

    raw = value.strip()

    if raw.startswith("data:"):
        _, b64 = raw.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    if raw.startswith("http://") or raw.startswith("https://"):
        req = Request(raw, headers={"User-Agent": "dinov3-endpoint/1.0"})
        with urlopen(req, timeout=30) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    try:
        return Image.open(io.BytesIO(base64.b64decode(raw, validate=True))).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("image must be URL, data-URL, or base64") from exc


def _pool_features(outputs) -> torch.Tensor:
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    # fallback: CLS token (index 0); DINOv3 also has register tokens after CLS
    return outputs.last_hidden_state[:, 0]


class EndpointHandler:
    def __init__(self, path: str = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        token = _hub_token()
        if token:
            # Явный login — иначе from_pretrained к gated facebook/* даёт 401
            try:
                from huggingface_hub import login

                login(token=token, add_to_git_credential=False)
            except Exception as exc:  # noqa: BLE001
                print(f"huggingface_hub.login warning: {exc}", flush=True)
        else:
            print(
                "WARN: HF_TOKEN / HUGGING_FACE_HUB_TOKEN не задан — "
                "gated facebook/dinov3 может не скачаться (401). "
                "Добавьте Secret HF_TOKEN в настройках Inference Endpoint.",
                flush=True,
            )

        # path = снапшот endpoint-repo. Без config.json грузим канонический Meta-чекпоинт.
        load_from = path
        cfg = Path(path) / "config.json" if path else None
        if not path or cfg is None or not cfg.exists():
            load_from = os.environ.get("DINO_MODEL_ID", DEFAULT_MODEL_ID)

        print(f"loading DINOv3 from={load_from} device={self.device}", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(
            load_from,
            token=token,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            load_from,
            torch_dtype=dtype,
            token=token,
        )
        self.model.to(self.device)
        self.model.eval()

        self.model_id = getattr(self.model.config, "_name_or_path", load_from) or load_from
        self.dim = int(getattr(self.model.config, "hidden_size", 0) or 0)

        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        inputs = self.processor(images=dummy, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = _pool_features(self.model(**inputs))
            if self.dim == 0:
                self.dim = int(feats.shape[-1])
        print(f"DINOv3 ready dim={self.dim}", flush=True)
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        t_total = time.perf_counter()

        payload = data.get("inputs", data)
        normalize = bool(data.get("normalize", True))

        images: list[Any] = []
        if isinstance(payload, str):
            images = [payload]
        elif isinstance(payload, list):
            images = payload
        elif isinstance(payload, dict):
            raw_images = payload.get("images") or payload.get("image")
            if raw_images is not None:
                images = raw_images if isinstance(raw_images, list) else [raw_images]
        else:
            raise ValueError("inputs must be str | list | dict with images")

        if not images:
            raise ValueError("provide inputs.images (DINOv3 — только image embeddings)")

        t0 = time.perf_counter()
        pil_images = [_load_image(im) for im in images]
        img_inputs = self.processor(images=pil_images, return_tensors="pt")
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        preprocess_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with torch.inference_mode():
            feats = _pool_features(self.model(**img_inputs)).float()
            if normalize:
                feats = _l2_normalize(feats)
        infer_ms = (time.perf_counter() - t0) * 1000

        return {
            "model_id": self.model_id,
            "dim": self.dim,
            "image_embeddings": feats.cpu().tolist(),
            "device": str(self.device),
            "timings_ms": {
                "preprocess": round(preprocess_ms, 1),
                "infer": round(infer_ms, 1),
                "total": round((time.perf_counter() - t_total) * 1000, 1),
            },
        }
