"""Локальная проверка EndpointHandler без деплоя endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from handler import EndpointHandler


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model_path", help="локальный путь или HF model id")
    p.add_argument("image", type=Path)
    args = p.parse_args()

    handler = EndpointHandler(path=args.model_path)
    b64 = base64.b64encode(args.image.read_bytes()).decode("ascii")
    payload = {
        "inputs": {"images": [f"data:image/jpeg;base64,{b64}"]},
        "normalize": True,
    }
    out = handler(payload)
    # не печатаем весь вектор
    summary = {
        "model_id": out["model_id"],
        "dim": out["dim"],
        "device": out["device"],
        "timings_ms": out["timings_ms"],
        "l2_sample": sum(x * x for x in out["image_embeddings"][0]) ** 0.5,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
