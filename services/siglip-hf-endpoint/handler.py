"""
Custom Inference Endpoint handler for SigLIP 2 image (and optional text) embeddings.

Place this file at the ROOT of your Hub model repo together with requirements.txt,
then create an Inference Endpoint from THAT repo (Task = Custom).

Request examples:

  {"inputs": {"images": ["data:image/jpeg;base64,..."]}}
  {"inputs": {"images": ["https://.../photo.jpg"]}}
  {"inputs": {"texts": ["красное сухое вино"]}}
  {"inputs": {"images": ["..."], "texts": ["..."]}, "normalize": true}

Response:

  {
    "model_id": "...",
    "dim": 768,
    "image_embeddings": [[...], ...],   # or null
    "text_embeddings": [[...], ...],    # or null
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
from transformers import AutoModel, AutoProcessor


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

    # data URL or bare base64
    if raw.startswith("data:"):
        _, b64 = raw.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    if raw.startswith("http://") or raw.startswith("https://"):
        req = Request(raw, headers={"User-Agent": "siglip2-endpoint/1.0"})
        with urlopen(req, timeout=30) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    # bare base64 (no data: prefix)
    try:
        return Image.open(io.BytesIO(base64.b64decode(raw, validate=True))).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("image must be URL, data-URL, or base64") from exc


class EndpointHandler:
    def __init__(self, path: str = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # path = локальный снапшот endpoint-repo. Если там нет весов (лёгкий deploy-repo),
        # грузим канонический чекпоинт Google с Hub.
        load_from = path
        cfg = Path(path) / "config.json" if path else None
        if not path or cfg is None or not cfg.exists():
            load_from = os.environ.get(
                "SIGLIP_MODEL_ID", "google/siglip2-base-patch16-384"
            )

        self.processor = AutoProcessor.from_pretrained(load_from, use_fast=False)
        self.model = AutoModel.from_pretrained(load_from, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

        self.model_id = getattr(self.model.config, "_name_or_path", path) or path
        self.dim = int(getattr(self.model.config, "projection_dim", 0) or 0)

        # warmup → узнать dim, если не в конфиге
        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        inputs = self.processor(images=dummy, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = self.model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            if self.dim == 0:
                self.dim = int(feats.shape[-1])

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        t_total = time.perf_counter()

        payload = data.get("inputs", data)
        normalize = bool(data.get("normalize", True))

        # допускаем короткий формат: {"inputs": "<base64>"} или {"inputs": ["..."]}
        images: list[Any] = []
        texts: list[str] = []

        if isinstance(payload, str):
            images = [payload]
        elif isinstance(payload, list):
            images = payload
        elif isinstance(payload, dict):
            raw_images = payload.get("images") or payload.get("image")
            raw_texts = payload.get("texts") or payload.get("text")
            if raw_images is not None:
                images = raw_images if isinstance(raw_images, list) else [raw_images]
            if raw_texts is not None:
                texts = raw_texts if isinstance(raw_texts, list) else [raw_texts]
        else:
            raise ValueError("inputs must be str | list | dict with images/texts")

        if not images and not texts:
            raise ValueError("provide at least one of inputs.images or inputs.texts")

        out: dict[str, Any] = {
            "model_id": self.model_id,
            "dim": self.dim,
            "image_embeddings": None,
            "text_embeddings": None,
            "device": str(self.device),
        }

        preprocess_ms = 0.0
        infer_ms = 0.0

        with torch.inference_mode():
            if images:
                t0 = time.perf_counter()
                pil_images = [_load_image(im) for im in images]
                img_inputs = self.processor(images=pil_images, return_tensors="pt")
                img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
                preprocess_ms += (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                feats = self.model.get_image_features(**img_inputs)
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                feats = feats.float()
                if normalize:
                    feats = _l2_normalize(feats)
                infer_ms += (time.perf_counter() - t0) * 1000
                out["image_embeddings"] = feats.cpu().tolist()

            if texts:
                t0 = time.perf_counter()
                txt_inputs = self.processor(
                    text=texts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}
                preprocess_ms += (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                feats = self.model.get_text_features(**txt_inputs)
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                feats = feats.float()
                if normalize:
                    feats = _l2_normalize(feats)
                infer_ms += (time.perf_counter() - t0) * 1000
                out["text_embeddings"] = feats.cpu().tolist()

        out["timings_ms"] = {
            "preprocess": round(preprocess_ms, 1),
            "infer": round(infer_ms, 1),
            "total": round((time.perf_counter() - t_total) * 1000, 1),
        }
        return out
