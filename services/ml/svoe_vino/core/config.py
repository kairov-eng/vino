"""Загрузка конфига пайплайна (YAML + env)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


def load_pipeline_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or Path(os.environ.get("PIPELINE_CONFIG", DEFAULT_CONFIG))
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    device = os.environ.get("DEVICE", cfg.get("device", "cpu"))
    cfg["device"] = device

    siglip = cfg.setdefault("siglip", {})
    if model := os.environ.get("SIGLIP_MODEL"):
        siglip["model_id"] = model
    elif device.startswith("cuda") and "so400m" not in str(siglip.get("model_id", "")):
        # опционально можно переключить на so400m вручную через SIGLIP_MODEL
        pass

    return cfg
