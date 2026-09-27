"""CLI: посчитать SigLIP 2 embedding для одного или нескольких изображений."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from svoe_vino.core.config import load_pipeline_config
from svoe_vino.pipeline.embed import build_siglip_embedder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SigLIP 2 image embedding (get_image_features)")
    parser.add_argument("images", nargs="+", type=Path, help="Пути к изображениям")
    parser.add_argument("--out", type=Path, default=None, help="Куда сохранить .npy (опционально)")
    parser.add_argument("--config", type=Path, default=None, help="configs/pipeline.yaml")
    args = parser.parse_args(argv)

    cfg = load_pipeline_config(args.config)
    embedder = build_siglip_embedder(cfg)

    images = [Image.open(p).convert("RGB") for p in args.images]
    vectors = embedder.embed(images)

    summary = {
        "model_id": embedder.model_id,
        "device": cfg.get("device"),
        "dim": embedder.dim,
        "count": int(vectors.shape[0]),
        "l2_norms": [float(np.linalg.norm(v)) for v in vectors],
        "files": [str(p) for p in args.images],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, vectors)
        print(f"saved: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
