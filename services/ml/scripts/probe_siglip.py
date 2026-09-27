"""Диагностика загрузки SigLIP 2 и get_image_features."""
from __future__ import annotations

import sys
import time
from pathlib import Path

print("1. imports...", flush=True)
t0 = time.perf_counter()
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

print(f"   ok in {time.perf_counter()-t0:.1f}s  torch={torch.__version__}", flush=True)

ckpt = "google/siglip2-base-patch16-384"
print(f"2. processor {ckpt}...", flush=True)
t0 = time.perf_counter()
processor = AutoProcessor.from_pretrained(ckpt, use_fast=False)
print(f"   ok in {time.perf_counter()-t0:.1f}s", flush=True)

print("3. model...", flush=True)
t0 = time.perf_counter()
model = AutoModel.from_pretrained(ckpt, dtype=torch.float32).eval()
print(f"   ok in {time.perf_counter()-t0:.1f}s", flush=True)

img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\dev\Vino2026\temp\unbound_sample.jpg")
print(f"4. embed {img_path}...", flush=True)
image = Image.open(img_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.inference_mode():
    feats = model.get_image_features(**inputs)
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)

print(f"   shape={tuple(feats.shape)} dtype={feats.dtype} l2={float(feats.norm()):.4f}", flush=True)
print("5. done", flush=True)
