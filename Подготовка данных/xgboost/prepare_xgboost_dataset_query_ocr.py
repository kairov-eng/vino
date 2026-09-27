r"""Датасет XGBoost: шумный OCR только в ЗАПРОСЕ, эталон всегда wines.label.

Алгоритм
--------
1. Берём search_photos с id >= --min-scan-id (по умолчанию 800).
2. Правила разметки как раньше:
   - manual_wines_id > matched; FP без manual → только hard-negative на найденном;
   - OCR запроса из скана: Google Vision всегда; OpenAI иначе Gemini.
3. Для каждой пары (скан, OCR-движок скана) строим группу:
   запрос = OCR скана, эталон кандидатов = wines.label (Gemini),
   кандидаты = SigLIP2 Top-20 этого поиска.
4. Если есть отобранное вино X — добавляем доп. группы ТОЛЬКО в train:
   запрос = label_ocr.ocr_google_vision / ocr_rapidocr_onnx вина X (sim < 1,
   текст ≠ label и ≠ уже взятых), эталон = те же wines.label,
   кандидаты = тот же Top-20: X → positive, остальные → negative.
5. Split по wine_id (по умолчанию 90/5/5). Все FP-группы принудительно в train.
   Validation/test — только OCR сканов (без catalog/synth).
6. Синтетика только в train (fp_synthetic):
   A twin-swap — подмешать токены близнеца из Top-20 (общий бренд/name);
   B shared-brand — этикетка без уникального name/grape;
   C garbage — цены/SKU, все кандидаты negative;
   D attr-conflict — цвет/сахар перевёрнут, все negative;
   + OCR-порча (trim/omoglyph/drop) как дырявые positives.
7. sample_weight: FP×3, twin/attr×3, brand/garbage×2.5, corrupt×1.5.
8. Признаки: text_features + hard_reject; cosine — только baseline (--drop-features).
9. Нормализация OCR: омоглифы + цифры-в-слове (как backend text_compare).
10. --exclude-listed-photos: не брать сканы, чей sha256 есть в
    xgb_exclude_photos (импорт папки: label_detect/19.import_xgb_exclude_folder.py).

Выход по умолчанию:
    C:\\dev\\Vino2026\\models\\data_xgboost_id800_fp_synth
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp_synthetic import (  # noqa: E402
    expand_fp_synthetics,
    force_fp_groups_to_train,
)
from prepare_dataset import (  # noqa: E402
    Candidate,
    load_wines,
    resolve_database_url,
)
from prepare_xgboost_synthetic import (  # noqa: E402
    DbScan,
    RealGroup,
    assign_split,
    choose_ocr_variants,
    deduplicate_scans,
    expand_real_groups,
    write_jsonl,
)
from text_features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    WineRef,
    group_features,
    normalize_text,
)

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE
REPO_ROOT = MODELS_DIR.parent
DEFAULT_ENV = HERE.parents[1] / "backend" / ".env"  # vino-svoe/backend/.env
DEFAULT_OUT = MODELS_DIR / "data_xgboost_id800_abs_v14"

CHANNEL_SECTIONS = {
    "catalog_google_vision": "ocr_google_vision",
    "catalog_rapidocr_onnx": "ocr_rapidocr_onnx",
}

COSINE_FEATURE_NAMES = [
    "siglip_cosine",
    "rel_cos_rank",
    "rel_cos_delta_top",
    "rel_cos_delta_mean",
    "rel_cos_z",
]

ID_COLUMNS = [
    "scan_id",
    "wine_id",
    "group_id",
    "split",
    "label",
    "label_kind",
    "ocr_engine",
    "data_source",
    "query_source",
    "original_scan_id",
    "base_group_id",
    "catalog_variant_sim",
    "sample_weight",
]


@dataclass
class CatalogOcr:
    channel: str
    text: str
    sim: float


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_exclude_sha256(engine) -> set[str]:
    """sha256 из xgb_exclude_photos (holdout-папки). Пусто, если таблицы нет."""
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text("select sha256 from xgb_exclude_photos where sha256 <> ''")
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Не удалось прочитать xgb_exclude_photos "
            f"({exc}). Сначала: alembic upgrade head "
            "и label_detect/19.import_xgb_exclude_folder.py"
        ) from exc
    return {str(r[0]).lower() for r in rows if r[0]}


def load_db_scans_from_id(engine, min_scan_id: int) -> list[DbScan]:
    sql = text(
        "select id, sha256, created_at, status, manual_wines_id "
        "from search_photos where id >= :min_id order by id"
    )
    with engine.connect() as connection:
        rows = connection.execute(sql, {"min_id": min_scan_id}).fetchall()

    from prepare_dataset import extract_eval_flags, extract_siglip_candidates

    parsed: list[DbScan] = []
    for row in rows:
        status = row[3] or {}
        ocr_variants = choose_ocr_variants(status)
        candidates = extract_siglip_candidates(status)
        if not ocr_variants or not candidates:
            continue
        manual = None
        try:
            manual = int(row[4]) if row[4] is not None else None
        except (TypeError, ValueError):
            manual = None
        matched = None
        try:
            if status.get("matched_wine_id") is not None:
                matched = int(status["matched_wine_id"])
        except (TypeError, ValueError):
            matched = None
        fp, fn = extract_eval_flags(status)

        if manual is not None:
            truth = manual
            fp_wrong = None
        elif fp:
            truth = None
            fp_wrong = matched
        else:
            truth = matched
            fp_wrong = None

        if truth is None and fp_wrong is None:
            continue
        candidate_ids = {c.wine_id for c in candidates}
        if truth is not None and truth not in candidate_ids:
            continue
        if fp_wrong is not None and fp_wrong not in candidate_ids:
            continue
        parsed.append(
            DbScan(
                scan_id=int(row[0]),
                sha256=str(row[1] or ""),
                created_at=row[2],
                status=status,
                manual_wine_id=manual,
                ocr_variants=ocr_variants,
                candidates=candidates,
                matched_wine_id=matched,
                false_positive=fp,
                false_negative=fn,
                truth_wine_id=truth,
                fp_wrong_wine_id=fp_wrong,
            )
        )
    return parsed


def load_catalog_query_ocr(
    engine,
    wine_ids: set[int],
    *,
    min_sim: dict[str, float],
) -> dict[int, list[CatalogOcr]]:
    """OCR каталога как текст ЗАПРОСА (sim < 1, текст ≠ wines.label)."""
    if not wine_ids:
        return {}
    out: dict[int, list[CatalogOcr]] = defaultdict(list)
    ids = sorted(wine_ids)
    with engine.connect() as connection:
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            rows = connection.execute(
                text("select id, label, label_ocr from wines where id = any(:ids)"),
                {"ids": chunk},
            ).fetchall()
            for wine_id, label, label_ocr in rows:
                base_key = normalize_text(str(label or ""))
                lo = label_ocr if isinstance(label_ocr, dict) else {}
                seen = {base_key} if base_key else set()
                for channel, section in CHANNEL_SECTIONS.items():
                    block = lo.get(section)
                    if not isinstance(block, dict):
                        continue
                    lines = block.get("lines")
                    if isinstance(lines, list) and lines:
                        text_value = "\n".join(str(x) for x in lines)
                    else:
                        text_value = str(block.get("text") or "")
                    key = normalize_text(text_value)
                    if len(key) < 3 or key in seen:
                        continue
                    compare = block.get("compare") if isinstance(block.get("compare"), dict) else {}
                    sim = _as_float(compare.get("sim"))
                    if sim is None or sim >= 1.0:
                        continue
                    floor = min_sim.get(channel, 0.0)
                    if sim < floor:
                        continue
                    seen.add(key)
                    out[int(wine_id)].append(
                        CatalogOcr(channel=channel, text=text_value, sim=sim)
                    )
    return dict(out)


def expand_catalog_query_groups(
    groups: list[RealGroup],
    catalog_ocr: dict[int, list[CatalogOcr]],
) -> tuple[list[RealGroup], Counter]:
    """Дубликаты групп: запрос = OCR каталога найденного вина; только train."""
    extra: list[RealGroup] = []
    skipped: Counter = Counter()
    next_id = max((g.sample_id for g in groups), default=0) + 1
    for group in groups:
        if group.split != "train":
            skipped["not_train"] += 1
            continue
        if group.truth_wine_id is None:
            skipped["no_positive"] += 1
            continue
        variants = catalog_ocr.get(group.truth_wine_id) or []
        if not variants:
            skipped["positive_without_catalog_ocr"] += 1
            continue
        for variant in variants:
            extra.append(
                replace(
                    group,
                    sample_id=next_id,
                    ocr_engine=variant.channel,
                    ocr_text=variant.text,
                )
            )
            # stash sim on object for row builder
            setattr(extra[-1], "_catalog_sim", variant.sim)
            setattr(extra[-1], "_query_source", "catalog_ocr")
            setattr(extra[-1], "_base_sample_id", group.sample_id)
            next_id += 1
    for group in groups:
        setattr(group, "_catalog_sim", None)
        setattr(group, "_query_source", "scan_ocr")
        setattr(group, "_base_sample_id", group.sample_id)
    return groups + extra, skipped


def build_rows_for_group(
    group: RealGroup,
    wines: dict[int, WineRef],
    reject_by_scan: dict[int, tuple[dict, set[int]]] | None = None,
) -> list[dict[str, Any]]:
    del reject_by_scan  # hard_reject больше не в фичах XGB
    refs: list[WineRef] = []
    candidates: list[Candidate] = []
    for candidate in group.candidates:
        ref = wines.get(candidate.wine_id)
        if ref is not None:
            refs.append(ref)
            candidates.append(candidate)
    if not refs:
        return []

    features = group_features(
        group.ocr_text,
        refs,
        [c.cosine for c in candidates],
    )

    query_source = getattr(group, "_query_source", "scan_ocr")
    catalog_sim = getattr(group, "_catalog_sim", None)
    base_id = getattr(group, "_base_sample_id", group.sample_id)
    data_source = getattr(group, "_data_source", "real")
    all_negative = bool(getattr(group, "_all_negative", False))
    synth_kind = getattr(group, "_synth_kind", None)
    sample_weight = float(getattr(group, "_sample_weight", 1.0) or 1.0)
    boost_neg = set(getattr(group, "_boost_neg_wine_ids", None) or ())

    rows: list[dict[str, Any]] = []
    for ref, candidate, feature_values in zip(refs, candidates, features):
        row_weight = sample_weight
        if all_negative:
            label = 0
            kind = str(synth_kind or "synth_all_neg")
        elif group.truth_wine_id is None:
            if candidate.wine_id != group.fp_wrong_wine_id:
                continue
            label = 0
            kind = "real_fp_hard"
            row_weight = max(sample_weight, 3.0)
        else:
            label = int(candidate.wine_id == group.truth_wine_id)
            if (not label) and candidate.wine_id in boost_neg:
                row_weight = max(row_weight, 4.0)
            if data_source == "synthetic":
                if query_source == "synth_twin":
                    kind = "synth_twin_tp" if label else "synth_twin_tn"
                elif query_source == "synth_brand":
                    kind = "synth_brand_tp" if label else "synth_brand_tn"
                elif query_source == "synth_corrupt":
                    kind = "synth_corrupt_tp" if label else "synth_corrupt_tn"
                else:
                    kind = "synthetic_tp" if label else "synthetic_tn"
            elif query_source == "catalog_ocr":
                kind = "catalog_tp" if label else "catalog_tn_embedding"
            else:
                kind = "real_tp" if label else "real_tn_embedding"

        row: dict[str, Any] = {
            "scan_id": group.sample_id,
            "wine_id": candidate.wine_id,
            "group_id": group.split_group_wine_id,
            "split": group.split,
            "label": label,
            "label_kind": kind,
            "ocr_engine": group.ocr_engine,
            "data_source": data_source,
            "query_source": query_source,
            "original_scan_id": group.original_scan_id,
            "base_group_id": base_id,
            "catalog_variant_sim": catalog_sim,
            "sample_weight": row_weight,
            "siglip_rank_original": candidate.rank,
        }
        row.update(feature_values)
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-url", default=None)
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-scan-id", type=int, default=800)
    p.add_argument("--min-sim-google-vision", type=float, default=0.0)
    p.add_argument("--min-sim-rapidocr", type=float, default=0.3)
    p.add_argument("--train", type=float, default=0.90)
    p.add_argument("--validation", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp-synth", action="store_true", help="не генерировать A–D синтетику")
    p.add_argument(
        "--exclude-listed-photos",
        action="store_true",
        help=(
            "не брать search_photos, чей sha256 есть в xgb_exclude_photos "
            "(файлы из holdout-папок, импорт: "
            "label_detect/19.import_xgb_exclude_folder.py)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine = create_engine(resolve_database_url(args.database_url, args.env_file))

    print(f"[1/6] Читаю search_photos id >= {args.min_scan_id} …")
    scans = load_db_scans_from_id(engine, args.min_scan_id)
    if args.exclude_listed_photos:
        exclude_hashes = load_exclude_sha256(engine)
        before = len(scans)
        scans = [
            s
            for s in scans
            if (s.sha256 or "").lower() not in exclude_hashes
        ]
        print(
            f"      exclude-listed-photos: hashes={len(exclude_hashes)}, "
            f"убрано сканов={before - len(scans)}, осталось={len(scans)}"
        )
    eligible = len(scans)
    scans, dropped = deduplicate_scans(scans)
    print(f"      пригодных={eligible}, после sha256-dedup={len(scans)} (-{dropped})")

    print("[2/6] Базовые группы (OCR скана) …")
    groups = expand_real_groups(scans)
    assign_split(groups, args.train, args.validation, args.seed)
    moved_fp = force_fp_groups_to_train(groups)
    split_counts = Counter(g.split for g in groups)
    print(f"      групп={len(groups)}, split={dict(split_counts)}, FP→train={moved_fp}")

    wine_ids: set[int] = set()
    for group in groups:
        wine_ids.add(group.split_group_wine_id)
        wine_ids.update(c.wine_id for c in group.candidates)
    truth_ids = {g.truth_wine_id for g in groups if g.truth_wine_id is not None}

    print("[3/6] OCR каталога как запрос (только train) …")
    min_sim = {
        "catalog_google_vision": args.min_sim_google_vision,
        "catalog_rapidocr_onnx": args.min_sim_rapidocr,
    }
    catalog_ocr = load_catalog_query_ocr(engine, truth_ids, min_sim=min_sim)
    n_with = sum(1 for wid in truth_ids if wid in catalog_ocr)
    by_ch = Counter(
        v.channel for variants in catalog_ocr.values() for v in variants
    )
    print(f"      вин с каталожным OCR-запросом: {n_with}/{len(truth_ids)}; каналы={dict(by_ch)}")
    all_groups, skipped = expand_catalog_query_groups(groups, catalog_ocr)
    n_catalog = sum(1 for g in all_groups if getattr(g, "_query_source", "") == "catalog_ocr")
    print(f"      добавлено catalog-query групп: {n_catalog}; пропуски={dict(skipped)}")

    print("[4/6] Эталоны wines.label + label_ocr attrs …")
    wines = load_wines(engine, wine_ids)
    print(f"      эталонов: {len(wines)}")

    synth_stats: Counter = Counter()
    if not args.no_fp_synth:
        print("[5/6] Синтетика FP A–D + OCR-порча (только train) …")
        synth_groups, synth_stats = expand_fp_synthetics(
            all_groups, wines, seed=args.seed
        )
        all_groups = all_groups + synth_groups
        print(f"      добавлено synth-групп: {len(synth_groups)}; {dict(synth_stats)}")
    else:
        print("[5/6] Синтетика FP пропущена (--no-fp-synth)")

    print("[6/6] Считаю текстовые признаки (v1.4 + siglip_cosine, OCR↔label) …")
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_audit: list[dict[str, Any]] = []
    for group in all_groups:
        built = build_rows_for_group(group, wines)
        if not built:
            continue
        rows_by_split[group.split].extend(built)
        group_audit.append(
            {
                "sample_id": group.sample_id,
                "original_scan_id": group.original_scan_id,
                "split": group.split,
                "query_source": getattr(group, "_query_source", "scan_ocr"),
                "data_source": getattr(group, "_data_source", "real"),
                "ocr_engine": group.ocr_engine,
                "truth_wine_id": group.truth_wine_id,
                "fp_wrong_wine_id": group.fp_wrong_wine_id,
                "catalog_variant_sim": getattr(group, "_catalog_sim", None),
                "sample_weight": getattr(group, "_sample_weight", 1.0),
                "all_negative": getattr(group, "_all_negative", False),
                "pair_count": len(built),
                "ocr_text": group.ocr_text,
                "synth_meta": getattr(group, "_synth_meta", None),
            }
        )

    out = args.out_dir
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    feature_cols = list(FEATURE_NAMES)

    stats: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        frame = pd.DataFrame(rows_by_split.get(split_name) or [])
        if frame.empty:
            frame = pd.DataFrame(
                columns=ID_COLUMNS + ["siglip_rank_original"] + feature_cols
            )
        else:
            if "sample_weight" not in frame.columns:
                frame["sample_weight"] = 1.0
            cols = [
                c
                for c in ID_COLUMNS + ["siglip_rank_original"] + feature_cols
                if c in frame.columns
            ]
            # уникальный порядок (siglip_cosine теперь в FEATURE_NAMES)
            seen: set[str] = set()
            uniq: list[str] = []
            for c in cols:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            frame = frame[uniq]
        path = out / "features" / f"{split_name}.parquet"
        frame.to_parquet(path, index=False)
        qs = (
            frame.groupby("query_source").size().to_dict()
            if "query_source" in frame.columns and len(frame)
            else {}
        )
        kinds = (
            frame.groupby("label_kind").size().to_dict()
            if "label_kind" in frame.columns and len(frame)
            else {}
        )
        stats[split_name] = {
            "pairs": int(len(frame)),
            "positives": int(frame["label"].sum()) if len(frame) else 0,
            "negatives": int((frame["label"] == 0).sum()) if len(frame) else 0,
            "query_sources": {str(k): int(v) for k, v in qs.items()},
            "label_kinds": {str(k): int(v) for k, v in kinds.items()},
            "mean_sample_weight": (
                float(frame["sample_weight"].mean()) if len(frame) else 0.0
            ),
        }
        print(
            f"      {split_name:<10} pairs={stats[split_name]['pairs']:<7} "
            f"pos={stats[split_name]['positives']:<5} "
            f"neg={stats[split_name]['negatives']:<7} "
            f"query={stats[split_name]['query_sources']}"
        )

    write_jsonl(out / "raw" / "groups.jsonl", group_audit)
    wines_dump = [
        {
            "wine_id": w.wine_id,
            "label": w.label,
            "name": w.name,
            "winery": w.winery,
            "category": w.category,
            "color": w.color,
            "grape": w.grape,
            "region": w.region,
        }
        for w in wines.values()
    ]
    write_jsonl(out / "raw" / "wines.jsonl", wines_dump)

    meta = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "min_scan_id": args.min_scan_id,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "excluded": ["rel_*", "hard_reject_*", "n_candidates"],
        "split_ratios": {
            "train": args.train,
            "validation": args.validation,
            "test": 1.0 - args.train - args.validation,
        },
        "fp_forced_to_train": moved_fp,
        "fp_synth_stats": dict(synth_stats),
        "algorithm": (
            "text features v1.4 + siglip_cosine; OCR↔wines.label "
            "(no CMS name/winery/grape/region); color/type from label_ocr; "
            "vintage+recency; digit-in-word+homoglyphs; no rel_*/hard_reject"
        ),
        "splits": stats,
        "catalog_query_skips": dict(skipped),
        "n_base_groups": len(groups),
        "n_all_groups": len(all_groups),
        "n_scans_after_dedup": len(scans),
    }
    (out / "dataset_stats.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_out = MODELS_DIR / "xgboost_text_matcher_abs_v14"
    print(f"      папка: {out}")
    print(
        "      обучение:\n"
        f"      python train_xgboost.py --data-dir {out} --out-dir {model_out}"
    )


if __name__ == "__main__":
    main()
