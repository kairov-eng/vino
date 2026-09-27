"""Глобальные визуальные эмбеддинги: SigLIP 2 (+ позже DINOv2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class Embedder(ABC):
    """Интерфейс: embed(images) -> np.ndarray[N, D], L2-нормированный."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, images: Sequence[Image.Image | np.ndarray]) -> np.ndarray: ...


def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class SigLIP2Embedder(Embedder):
    """Image embeddings через SigLIP 2 `get_image_features` + L2-norm."""

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-384",
        device: str = "cpu",
        batch_size: int = 8,
        warmup: bool = True,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        self._model_id = model_id
        self.device = torch.device(device)
        self.batch_size = batch_size
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id, dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()

        # размерность узнаём на dummy-прогоне или из конфига projection
        self._dim = int(getattr(self.model.config, "projection_dim", 0) or 0)
        if warmup or self._dim == 0:
            self._dim = self._warmup()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _warmup(self) -> int:
        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        vec = self.embed([dummy])
        return int(vec.shape[-1])

    @torch.inference_mode()
    def embed(self, images: Sequence[Image.Image | np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim or 0), dtype=np.float32)

        pil_images = [_to_pil(im) for im in images]
        out_chunks: list[np.ndarray] = []

        for start in range(0, len(pil_images), self.batch_size):
            batch = pil_images[start : start + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            feats = self.model.get_image_features(**inputs)
            # на случай, если вернётся BaseModelOutputWithPooling
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0]

            feats = _l2_normalize(feats.float())
            out_chunks.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(out_chunks, axis=0)


def build_siglip_embedder(cfg: dict) -> SigLIP2Embedder:
    sig = cfg.get("siglip", {})
    return SigLIP2Embedder(
        model_id=sig.get("model_id", "google/siglip2-base-patch16-384"),
        device=cfg.get("device", "cpu"),
        batch_size=int(sig.get("batch_size", 8)),
        warmup=bool(sig.get("warmup", True)),
    )
