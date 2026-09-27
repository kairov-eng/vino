r"""XGBoost dataset v2: реальные сканы + каналы OCR каталога + синтетика запроса.

Пишет отдельный набор в C:\dev\Vino2026\models\data_xgboost_v2 и не трогает
предыдущие наборы (models/data, models/data_xgboost_synthetic_40_60).

================================ АЛГОРИТМ ================================

ШАГ 1. Сканы поиска (search_photos), правила прежние
  - истинное вино: manual_wines_id в приоритете, иначе status.matched_wine_id;
  - false_positive=1 без manual: найденное вино только как hard negative,
    остальные кандидаты не размечаются (истинное неизвестно);
  - отдельного класса false negative нет;
  - кандидаты: SigLIP2 Top-20 из steps.candidates.siglip2 вместе с cosine;
  - OCR запроса: Google Vision + OpenAI дают два варианта; Gemini берётся
    только при отсутствии OpenAI; одинаковые тексты схлопываются;
  - повторные сканы одного фото схлопываются по sha256
    (manual > false_positive > обычный > более новый).

ШАГ 2. Каналы эталонного текста каталога (wines.label_ocr)
  - базовый канал gemini = wines.label (совпадает с label_ocr.full_text);
  - канал google_vision = label_ocr.ocr_google_vision.text;
  - канал rapidocr_onnx = label_ocr.ocr_rapidocr_onnx.text.
  Вариант пригоден, если: секция есть, текст непустой, compare.sim задан и
  sim < 1 (при sim = 1 текст совпал с Gemini и не несёт нового), sim >= порога
  канала (RapidOCR по умолчанию 0.3, ниже идёт нечитаемый шум), и текст
  отличается от базового. Совпавшие между собой варианты схлопываются.
  Подменяется ТОЛЬКО текст этикетки. Поля карточки (name, winery, category,
  grape_variety, region) приходят из каталога, а не из OCR, и не меняются.

ШАГ 3. Split по wine_id (70/15/15) до генерации пар. Дубликаты каналов
  наследуют split базовой группы, поэтому одно вино не попадает в train и test
  одновременно.

ШАГ 4. Базовые группы: одна группа = один OCR-текст запроса x 20 кандидатов.
  Истинное вино label=1, остальные кандидаты label=0.

ШАГ 5. Удвоение через каналы каталога (по умолчанию только train)
  Для каждой базовой группы и каждого канала создаётся дубликат, если у
  ПОЗИТИВНОГО вина этой группы есть пригодный вариант канала. Внутри дубликата
  каждый кандидат берёт свой текст того же канала, а при его отсутствии
  откатывается на базовый wines.label: это имитирует «каталог, собранный
  движком X», как в продакшене, где каталог построен одним пайплайном.
  Дубликат с полностью совпавшим набором эталонных текстов отбрасывается.
  Строки, не изменившиеся относительно базы, помечаются row_same_as_base.

ШАГ 6. Признаки hard reject считаются внутри этого же скрипта (отдельного
  постобработчика больше нет): name, winery, grape, color, type,
  exclusive_lexicon, label_lines с конкретной причиной, HSV hard reject.
  Привязка по (scan_id, wine_id, OCR engine); от канала каталога не зависят,
  поэтому у дубликатов те же значения.

ШАГ 7. Синтетика на стороне ЗАПРОСА: обрезка букв в начале/конце слова, замена
  на визуально похожую латиницу и обратно, удаление малозначимых слов, удаление
  одной характеристики (год / цвет-тип / сорт / регион), 2-4 операции, результат
  обязан быть уникальным. Варианты каталога считаются РЕАЛЬНЫМИ данными, поэтому
  цель 40% real / 60% synthetic пересчитывается от нового объёма train. Базы для
  порчи берутся из всех реальных train-групп, включая дубликаты каналов.
  Validation и test остаются полностью реальными и на базовом канале каталога.

ШАГ 8. Запись features/{train,validation,test}.parquet, аудита в raw/ и
  dataset_stats.json. Служебные колонки catalog_ocr_channel,
  catalog_variant_sim, base_group_id, row_same_as_base нужны для анализа и НЕ
  подаются в модель: в продакшене канал каталога всегда один.

================================= ЗАПУСК =================================

    cd C:\dev\Vino2026\models\ocr_matcher
    python prepare_xgboost_dataset_v2.py
    python prepare_xgboost_dataset_v2.py --limit-wines 10     # быстрая проверка
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hard_reject_features import (  # noqa: E402
    HARD_REJECT_FEATURE_NAMES,
    HARD_REJECT_FEATURE_VERSION,
    build_reject_index,
    hard_reject_features,
)
from prepare_dataset import Candidate, resolve_database_url  # noqa: E402
from prepare_xgboost_synthetic import (  # noqa: E402
    DbScan,
    corrupt_ocr,
    deduplicate_scans,
    load_db_scans,
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
DEFAULT_OUT = MODELS_DIR / "data_xgboost_v2"

BASE_CHANNEL = "gemini"
CHANNEL_SECTIONS = {
    "google_vision": "ocr_google_vision",
    "rapidocr_onnx": "ocr_rapidocr_onnx",
}

ID_COLUMNS = [
    "scan_id",
    "wine_id",
    "group_id",
    "split",
    "label",
    "label_kind",
    "ocr_engine",
    "data_source",
    "original_scan_id",
    "siglip_rank_original",
    "catalog_ocr_channel",
    "catalog_variant_sim",
    "base_group_id",
    "row_same_as_base",
]


@dataclass
class CatalogWine:
    """Эталон вина: базовая карточка + альтернативные тексты этикетки."""

    ref: WineRef
    variants: dict[str, tuple[str, float]] = field(default_factory=dict)


@dataclass
class Group:
    """Одна группа = один текст запроса против Top-20 кандидатов SigLIP2."""

    group_key: int
    base_group_key: int
    original_scan_id: int
    split_wine_id: int
    truth_wine_id: int | None
    fp_wrong_wine_id: int | None
    ocr_engine: str
    ocr_text: str
    candidates: list[Candidate]
    catalog_channel: str
    data_source: str
    split: str | None = None
    synthetic_meta: dict[str, Any] | None = None


def load_catalog(
    engine,
    wine_ids: set[int],
    *,
    channels: list[str],
    min_sim: dict[str, float],
) -> dict[int, CatalogWine]:
    """Карточки вин + пригодные варианты текста этикетки по каналам."""
    out: dict[int, CatalogWine] = {}
    ids = sorted(wine_ids)
    with engine.connect() as connection:
        for offset in range(0, len(ids), 500):
            chunk = ids[offset : offset + 500]
            rows = connection.execute(
                text(
                    "select id, label, name, winery, category, color, "
                    "grape_variety, region, label_ocr "
                    "from wines where id = any(:ids)"
                ),
                {"ids": chunk},
            ).fetchall()
            for row in rows:
                ref = WineRef(
                    wine_id=int(row[0]),
                    label=row[1],
                    name=row[2],
                    winery=row[3],
                    category=row[4],
                    color=row[5],
                    grape=row[6],
                    region=row[7],
                )
                label_ocr = row[8] if isinstance(row[8], dict) else {}
                variants: dict[str, tuple[str, float]] = {}
                seen = {normalize_text(ref.ref_text())}
                for channel in channels:
                    section = label_ocr.get(CHANNEL_SECTIONS[channel])
                    if not isinstance(section, dict):
                        continue
                    raw = str(section.get("text") or "").strip()
                    if not raw:
                        raw = "\n".join(
                            str(x).strip()
                            for x in (section.get("lines") or [])
                            if str(x).strip()
                        )
                    if not raw.strip():
                        continue
                    try:
                        sim = float((section.get("compare") or {}).get("sim"))
                    except (TypeError, ValueError):
                        continue
                    # sim = 1 → текст совпал с Gemini, нового сигнала нет.
                    if sim >= 1.0 or sim < min_sim.get(channel, 0.0):
                        continue
                    key = normalize_text(raw)
                    if key in seen:
                        continue
                    seen.add(key)
                    variants[channel] = (raw, sim)
                out[int(row[0])] = CatalogWine(ref=ref, variants=variants)
    return out


def expand_base_groups(scans: list[DbScan]) -> list[Group]:
    groups: list[Group] = []
    key = 1
    for scan in scans:
        anchor = scan.truth_wine_id or scan.fp_wrong_wine_id
        if anchor is None:
            continue
        for engine_name, ocr_text in scan.ocr_variants:
            groups.append(
                Group(
                    group_key=key,
                    base_group_key=key,
                    original_scan_id=scan.scan_id,
                    split_wine_id=int(anchor),
                    truth_wine_id=scan.truth_wine_id,
                    fp_wrong_wine_id=scan.fp_wrong_wine_id,
                    ocr_engine=engine_name,
                    ocr_text=ocr_text,
                    candidates=scan.candidates,
                    catalog_channel=BASE_CHANNEL,
                    data_source="real",
                )
            )
            key += 1
    return groups


def assign_split(
    groups: list[Group], train_ratio: float, val_ratio: float, seed: int
) -> None:
    """Split по wine_id с приблизительным балансом по числу групп."""
    by_wine: dict[int, list[Group]] = defaultdict(list)
    for group in groups:
        by_wine[group.split_wine_id].append(group)
    wine_ids = list(by_wine)
    random.Random(seed).shuffle(wine_ids)
    total = len(groups)
    seen = 0
    for wine_id in wine_ids:
        fraction = seen / total if total else 0.0
        if fraction < train_ratio:
            split = "train"
        elif fraction < train_ratio + val_ratio:
            split = "validation"
        else:
            split = "test"
        for group in by_wine[wine_id]:
            group.split = split
        seen += len(by_wine[wine_id])


def _label_for_channel(
    catalog_wine: CatalogWine, channel: str
) -> tuple[str | None, float, bool]:
    """(текст этикетки, sim варианта, совпадает ли с базой)."""
    base_label = catalog_wine.ref.label
    if channel == BASE_CHANNEL:
        return base_label, float("nan"), True
    variant = catalog_wine.variants.get(channel)
    if variant is None:
        return base_label, float("nan"), True
    return variant[0], variant[1], False


def build_rows(
    group: Group,
    catalog: dict[int, CatalogWine],
    reject_index: dict[str, dict[int, dict[str, Any]]],
    hsv_ids: set[int],
) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    """Строки признаков группы + подпись набора эталонов для дедупликации."""
    refs: list[WineRef] = []
    candidates: list[Candidate] = []
    sims: list[float] = []
    same_flags: list[bool] = []
    signature: list[tuple[int, str]] = []

    for candidate in group.candidates:
        catalog_wine = catalog.get(candidate.wine_id)
        if catalog_wine is None:
            continue
        label, sim, same = _label_for_channel(catalog_wine, group.catalog_channel)
        refs.append(replace(catalog_wine.ref, label=label))
        candidates.append(candidate)
        sims.append(sim)
        same_flags.append(same)
        signature.append((candidate.wine_id, normalize_text(label or "")))

    if not refs:
        return [], []

    features = group_features(
        group.ocr_text, refs, [candidate.cosine for candidate in candidates]
    )
    engine_label = (
        f"synthetic_{group.ocr_engine}"
        if group.data_source == "synthetic"
        else group.ocr_engine
    )

    rows: list[dict[str, Any]] = []
    for candidate, values, sim, same in zip(candidates, features, sims, same_flags):
        if group.truth_wine_id is None:
            if candidate.wine_id != group.fp_wrong_wine_id:
                continue
            label_value, kind = 0, "real_fp_hard"
        else:
            label_value = int(candidate.wine_id == group.truth_wine_id)
            if group.data_source == "synthetic":
                kind = "synthetic_tp_hard" if label_value else "synthetic_tn_embedding"
            else:
                kind = "real_tp" if label_value else "real_tn_embedding"

        row: dict[str, Any] = {
            "scan_id": group.group_key,
            "wine_id": candidate.wine_id,
            "group_id": group.split_wine_id,
            "split": group.split,
            "label": label_value,
            "label_kind": kind,
            "ocr_engine": engine_label,
            "data_source": group.data_source,
            "original_scan_id": group.original_scan_id,
            "siglip_rank_original": candidate.rank,
            "catalog_ocr_channel": group.catalog_channel,
            "catalog_variant_sim": sim,
            "base_group_id": group.base_group_key,
            "row_same_as_base": int(same),
        }
        row.update(values)
        row.update(
            hard_reject_features(
                reject_index,
                hsv_ids,
                wine_id=candidate.wine_id,
                ocr_engine=group.ocr_engine,
            )
        )
        rows.append(row)
    return rows, signature


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--channels", default="google_vision,rapidocr_onnx")
    parser.add_argument("--min-sim-google-vision", type=float, default=0.0)
    parser.add_argument("--min-sim-rapidocr", type=float, default=0.3)
    parser.add_argument(
        "--variant-splits",
        default="train",
        help="где создавать дубликаты каналов каталога (train|train,validation,test)",
    )
    parser.add_argument("--synthetic-ratio", type=float, default=0.60)
    parser.add_argument("--max-synthetic-per-base", type=int, default=4)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-scans", type=int, default=None)
    parser.add_argument(
        "--limit-wines",
        type=int,
        default=None,
        help="оставить только N первых вин-якорей (быстрая проверка пайплайна)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.synthetic_ratio < 1.0:
        raise SystemExit("--synthetic-ratio должен быть между 0 и 1")
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    unknown = [c for c in channels if c not in CHANNEL_SECTIONS]
    if unknown:
        raise SystemExit(f"Неизвестные каналы: {unknown}")
    min_sim = {
        "google_vision": args.min_sim_google_vision,
        "rapidocr_onnx": args.min_sim_rapidocr,
    }
    variant_splits = {s.strip() for s in args.variant_splits.split(",") if s.strip()}

    db_url = resolve_database_url(args.database_url, args.env_file)
    engine = create_engine(db_url)

    print("[1/8] Читаю search_photos (manual > matched, FP, правила OCR) …")
    scans = load_db_scans(engine, args.limit_scans)
    eligible = len(scans)
    scans, dropped = deduplicate_scans(scans)
    print(f"      пригодных={eligible}, после sha256-dedup={len(scans)} (-{dropped})")
    if args.limit_wines:
        anchors: list[int] = []
        for scan in scans:
            anchor = scan.truth_wine_id or scan.fp_wrong_wine_id
            if anchor is not None and anchor not in anchors:
                anchors.append(int(anchor))
            if len(anchors) >= args.limit_wines:
                break
        keep = set(anchors)
        scans = [
            s
            for s in scans
            if (s.truth_wine_id or s.fp_wrong_wine_id) in keep
        ]
        print(f"      ограничение --limit-wines: {len(keep)} вин, {len(scans)} сканов")

    print("[2/8] Индексирую причины hard reject …")
    reject_by_scan: dict[int, tuple[dict[str, dict[int, dict[str, Any]]], set[int]]] = {}
    for scan in scans:
        reject_by_scan[scan.scan_id] = build_reject_index(scan.status)
        scan.status = {}  # статусы тяжёлые, дальше не нужны

    groups = expand_base_groups(scans)
    print(f"[3/8] Базовых реальных групп: {len(groups)}")

    assign_split(groups, args.train, args.validation, args.seed)
    print("      split:", dict(Counter(g.split for g in groups)))

    wine_ids: set[int] = set()
    for group in groups:
        wine_ids.update(c.wine_id for c in group.candidates)
        for extra in (group.truth_wine_id, group.fp_wrong_wine_id):
            if extra is not None:
                wine_ids.add(int(extra))
    catalog = load_catalog(engine, wine_ids, channels=channels, min_sim=min_sim)
    with_variant = {
        channel: sum(1 for w in catalog.values() if channel in w.variants)
        for channel in channels
    }
    print(f"[4/8] Эталонов вин: {len(catalog)}; с вариантами: {with_variant}")

    print("[5/8] Дубликаты групп по каналам каталога …")
    next_key = max((g.group_key for g in groups), default=0) + 1
    variant_groups: list[Group] = []
    skipped_variants: Counter[str] = Counter()
    for group in list(groups):
        if group.split not in variant_splits:
            continue
        if group.truth_wine_id is None:
            skipped_variants["no_positive"] += 1
            continue
        positive = catalog.get(int(group.truth_wine_id))
        if positive is None:
            continue
        for channel in channels:
            if channel not in positive.variants:
                skipped_variants[f"{channel}:positive_without_variant"] += 1
                continue
            variant_groups.append(
                replace(
                    group,
                    group_key=next_key,
                    base_group_key=group.group_key,
                    catalog_channel=channel,
                )
            )
            next_key += 1
    print(
        f"      создано дубликатов: {len(variant_groups)}; "
        f"пропуски: {dict(skipped_variants)}"
    )

    print("[6/8] Считаю признаки реальных групп …")
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit: list[dict[str, Any]] = []
    signatures: dict[int, list[tuple[int, str]]] = {}
    real_train_groups: list[Group] = []
    dropped_identical = 0

    for group in groups + variant_groups:
        index, hsv_ids = reject_by_scan.get(
            group.original_scan_id, ({"_final": {}}, set())
        )
        rows, signature = build_rows(group, catalog, index, hsv_ids)
        if not rows:
            continue
        if group.catalog_channel != BASE_CHANNEL:
            if signatures.get(group.base_group_key) == signature:
                dropped_identical += 1
                continue
        else:
            signatures[group.group_key] = signature
        rows_by_split[group.split or "train"].extend(rows)
        audit.append(
            {
                "group_key": group.group_key,
                "base_group_key": group.base_group_key,
                "original_scan_id": group.original_scan_id,
                "split": group.split,
                "catalog_ocr_channel": group.catalog_channel,
                "ocr_engine": group.ocr_engine,
                "truth_wine_id": group.truth_wine_id,
                "fp_wrong_wine_id": group.fp_wrong_wine_id,
                "pair_count": len(rows),
                "ocr_text": group.ocr_text,
            }
        )
        if group.split == "train" and group.truth_wine_id is not None:
            real_train_groups.append(group)
    if dropped_identical:
        print(f"      отброшено дубликатов без отличий: {dropped_identical}")

    real_train_rows = len(rows_by_split["train"])
    target_synthetic = round(
        real_train_rows * args.synthetic_ratio / (1.0 - args.synthetic_ratio)
    )
    print(
        f"[7/8] Синтетика запроса: real train={real_train_rows}, "
        f"цель synthetic={target_synthetic}"
    )

    used_texts = {normalize_text(g.ocr_text) for g in groups if g.ocr_text.strip()}
    rng = random.Random(args.seed)
    base_order = real_train_groups[:]
    rng.shuffle(base_order)
    synthetic_rows: list[dict[str, Any]] = []
    synthetic_audit: list[dict[str, Any]] = []
    per_base: Counter[int] = Counter()
    exhausted: set[int] = set()
    skipped_synth: Counter[str] = Counter()
    cycle = 0

    while base_order and len(synthetic_rows) < target_synthetic:
        base = base_order[cycle % len(base_order)]
        cycle += 1
        if (
            per_base[base.group_key] >= args.max_synthetic_per_base
            or base.group_key in exhausted
        ):
            if all(
                per_base[g.group_key] >= args.max_synthetic_per_base
                or g.group_key in exhausted
                for g in base_order
            ):
                break
            continue
        positive = catalog.get(int(base.truth_wine_id or -1))
        if positive is None:
            exhausted.add(base.group_key)
            continue
        label, _, _ = _label_for_channel(positive, base.catalog_channel)
        ref = replace(positive.ref, label=label)
        try:
            synthetic_text, operations, severity = corrupt_ocr(
                base.ocr_text,
                ref,
                args.seed + base.group_key,
                per_base[base.group_key],
                used_texts,
            )
        except RuntimeError:
            exhausted.add(base.group_key)
            skipped_synth["not_enough_text"] += 1
            continue

        synthetic_group = replace(
            base,
            group_key=next_key,
            base_group_key=base.group_key,
            data_source="synthetic",
            ocr_text=synthetic_text,
        )
        index, hsv_ids = reject_by_scan.get(
            base.original_scan_id, ({"_final": {}}, set())
        )
        rows, _ = build_rows(synthetic_group, catalog, index, hsv_ids)
        if not rows:
            exhausted.add(base.group_key)
            continue
        synthetic_rows.extend(rows)
        synthetic_audit.append(
            {
                "group_key": next_key,
                "base_group_key": base.group_key,
                "original_scan_id": base.original_scan_id,
                "catalog_ocr_channel": base.catalog_channel,
                "ocr_engine": base.ocr_engine,
                "severity": severity,
                "operations": operations,
                "original_ocr_text": base.ocr_text,
                "ocr_text": synthetic_text,
            }
        )
        per_base[base.group_key] += 1
        next_key += 1

    rows_by_split["train"].extend(synthetic_rows)

    print("[8/8] Записываю набор …")
    out = args.out_dir
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    feature_columns = FEATURE_NAMES + HARD_REJECT_FEATURE_NAMES
    split_stats: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        rows = rows_by_split[split]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame[ID_COLUMNS + feature_columns]
        frame.to_parquet(out / "features" / f"{split}.parquet", index=False)
        sources = Counter(r["data_source"] for r in rows)
        split_stats[split] = {
            "pairs": len(rows),
            "real_pairs": sources.get("real", 0),
            "synthetic_pairs": sources.get("synthetic", 0),
            "synthetic_fraction": (
                sources.get("synthetic", 0) / len(rows) if rows else 0.0
            ),
            "positives": sum(int(r["label"]) for r in rows),
            "negatives": sum(1 - int(r["label"]) for r in rows),
            "groups": len({r["scan_id"] for r in rows}),
            "label_kinds": dict(Counter(r["label_kind"] for r in rows)),
            "catalog_channels": dict(
                Counter(r["catalog_ocr_channel"] for r in rows)
            ),
            "rows_same_as_base": sum(int(r["row_same_as_base"]) for r in rows),
        }

    write_jsonl(out / "raw" / "groups.jsonl", audit)
    write_jsonl(out / "raw" / "synthetic_variants.jsonl", synthetic_audit)

    stats = {
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
        "dataset_version": "v2",
        "feature_version": FEATURE_VERSION,
        "hard_reject_feature_version": HARD_REJECT_FEATURE_VERSION,
        "feature_count": len(feature_columns),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k != "database_url"
        },
        "policy": {
            "positive_priority": [
                "search_photos.manual_wines_id",
                "status.matched_wine_id",
            ],
            "false_positive_without_manual": "matched wine only -> hard negative",
            "false_negative": "not a class",
            "query_ocr": "google_vision + openai; gemini only when openai absent",
            "catalog_channels": channels,
            "catalog_min_sim": min_sim,
            "catalog_variant_rule": (
                "sim < 1, text non-empty, differs from base; whole group switches "
                "channel with fallback to wines.label"
            ),
            "variant_splits": sorted(variant_splits),
            "synthetic_scope": "train only; query-side corruption",
        },
        "db_scans_eligible": eligible,
        "db_scans_after_dedup": len(scans),
        "base_real_groups": len(groups),
        "variant_groups": len(variant_groups),
        "variant_groups_dropped_identical": dropped_identical,
        "variant_skips": dict(skipped_variants),
        "catalog_wines": len(catalog),
        "catalog_wines_with_variant": with_variant,
        "synthetic_groups": len(synthetic_audit),
        "synthetic_severity": dict(
            Counter(r["severity"] for r in synthetic_audit)
        ),
        "synthetic_operations": dict(
            Counter(
                op.split(":", 1)[0]
                for r in synthetic_audit
                for op in r["operations"]
            )
        ),
        "synthetic_skips": dict(skipped_synth),
        "splits": split_stats,
    }
    (out / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for split, stat in split_stats.items():
        print(
            f"      {split:<10} pairs={stat['pairs']:<7} real={stat['real_pairs']:<7} "
            f"synth={stat['synthetic_pairs']:<7} synth%={stat['synthetic_fraction']:.2%} "
            f"pos={stat['positives']:<5} neg={stat['negatives']:<7} "
            f"channels={stat['catalog_channels']}"
        )
    empty = [s for s, stat in split_stats.items() if stat["pairs"] == 0]
    if empty:
        print(
            f"      ВНИМАНИЕ: пустые сплиты {empty} — слишком мало вин "
            f"для деления {args.train}/{args.validation}"
        )
    print(f"      папка: {out}")


if __name__ == "__main__":
    main()
