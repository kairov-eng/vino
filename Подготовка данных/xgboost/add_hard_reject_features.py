"""Добавить признаки hard reject в существующие parquet без изменения строк.

Использование:
    python add_hard_reject_features.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hard_reject_features import (  # noqa: E402
    HARD_REJECT_FEATURE_NAMES,
    HARD_REJECT_FEATURE_VERSION,
    build_reject_index,
    hard_reject_features,
)
from prepare_dataset import resolve_database_url  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE
REPO_ROOT = MODELS_DIR.parent
DEFAULT_ENV = HERE.parents[1] / "backend" / ".env"  # vino-svoe/backend/.env
DEFAULT_DATA = MODELS_DIR / "data_xgboost_synthetic_40_60"


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic_path = args.data_dir / "raw" / "synthetic_variants.jsonl"
    strings_hash_before = file_sha256(synthetic_path)

    frames: dict[str, pd.DataFrame] = {}
    scan_ids: set[int] = set()
    identity_before: dict[str, list[tuple]] = {}
    for split in ("train", "validation", "test"):
        path = args.data_dir / "features" / f"{split}.parquet"
        if not path.exists():
            raise SystemExit(f"Нет {path}")
        frame = pd.read_parquet(path)
        required = {"original_scan_id", "wine_id", "ocr_engine"}
        if not required.issubset(frame.columns):
            raise SystemExit(f"{path}: нет колонок {sorted(required - set(frame.columns))}")
        frames[split] = frame
        scan_ids.update(frame["original_scan_id"].astype(int).unique().tolist())
        identity_before[split] = list(
            frame[
                [
                    "scan_id",
                    "wine_id",
                    "label",
                    "ocr_engine",
                    "data_source",
                    "original_scan_id",
                ]
            ].itertuples(index=False, name=None)
        )

    db_url = resolve_database_url(args.database_url, args.env_file)
    engine = create_engine(db_url)
    statuses: dict[int, dict] = {}
    ids = sorted(scan_ids)
    with engine.connect() as connection:
        for offset in range(0, len(ids), 500):
            chunk = ids[offset : offset + 500]
            rows = connection.execute(
                text("select id, status from search_photos where id = any(:ids)"),
                {"ids": chunk},
            ).fetchall()
            statuses.update({int(row[0]): row[1] or {} for row in rows})

    indexes = {
        scan_id: build_reject_index(status)
        for scan_id, status in statuses.items()
    }
    counters: Counter[str] = Counter()
    for split, frame in frames.items():
        feature_rows: list[dict[str, float]] = []
        for row in frame.itertuples(index=False):
            reject_index, hsv_ids = indexes.get(
                int(row.original_scan_id), ({"_final": {}}, set())
            )
            values = hard_reject_features(
                reject_index,
                hsv_ids,
                wine_id=int(row.wine_id),
                ocr_engine=str(row.ocr_engine),
            )
            feature_rows.append(values)
            if values["hard_reject_detail_available"]:
                counters[f"{split}.detail_available"] += 1
            if values["hard_reject_any"]:
                counters[f"{split}.hard_any"] += 1
            if values["hard_reject_hsv"]:
                counters[f"{split}.hsv"] += 1

        additions = pd.DataFrame(feature_rows, index=frame.index)
        for name in HARD_REJECT_FEATURE_NAMES:
            frame[name] = additions[name].astype(float)

        # Гарантия «те же строки»: идентичность и порядок строк неизменны.
        identity_after = list(
            frame[
                [
                    "scan_id",
                    "wine_id",
                    "label",
                    "ocr_engine",
                    "data_source",
                    "original_scan_id",
                ]
            ].itertuples(index=False, name=None)
        )
        if identity_after != identity_before[split]:
            raise RuntimeError(f"Изменились строки/порядок в split={split}")
        frame.to_parquet(
            args.data_dir / "features" / f"{split}.parquet", index=False
        )

    strings_hash_after = file_sha256(synthetic_path)
    if strings_hash_before != strings_hash_after:
        raise RuntimeError("Изменился synthetic_variants.jsonl")

    report = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "hard_reject_feature_version": HARD_REJECT_FEATURE_VERSION,
        "feature_names": HARD_REJECT_FEATURE_NAMES,
        "source_scans": len(statuses),
        "synthetic_strings_sha256_before": strings_hash_before,
        "synthetic_strings_sha256_after": strings_hash_after,
        "strings_unchanged": strings_hash_before == strings_hash_after,
        "counts": dict(counters),
    }
    (args.data_dir / "hard_reject_augmentation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
