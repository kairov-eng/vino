"""Замер latency SigLIP 2: load + один вектор."""
from __future__ import annotations

import time

from PIL import Image

from svoe_vino.core.config import load_pipeline_config
from svoe_vino.pipeline.embed import build_siglip_embedder


def main() -> None:
    img = Image.open(r"C:\dev\Vino2026\temp\unbound_sample.jpg").convert("RGB")
    cfg = load_pipeline_config()
    sig = {**cfg.get("siglip", {}), "warmup": False}

    t0 = time.perf_counter()
    emb = build_siglip_embedder({**cfg, "siglip": sig})
    load_s = time.perf_counter() - t0

    emb.embed([img])  # warmup

    times: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        emb.embed([img])
        times.append(time.perf_counter() - t0)

    print(f"device={cfg.get('device')} model={emb.model_id} dim={emb.dim}")
    print(f"load_sec={load_s:.2f}")
    print(f"infer_sec={[round(x, 3) for x in times]}")
    print(f"infer_mean_sec={sum(times) / len(times):.3f}")
    print(f"infer_min_sec={min(times):.3f}")


if __name__ == "__main__":
    main()
