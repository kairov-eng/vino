"""Отдельный датасет XGBoost: реальные OCR + 60% синтетических OCR.

Правила:
- positive wine: search_photos.manual_wines_id (приоритет), иначе matched_wine_id;
- false_positive без manual: найденное вино — только hard negative;
- false_negative не является отдельным классом; синтетические сложные positives
  имитируют причины будущих FN, но сохраняют label=1;
- negatives: остальные кандидаты исходного SigLIP2 Top-20;
- OCR-варианты:
    Google Vision + OpenAI -> оба;
    OpenAI + Gemini -> только OpenAI (Gemini подавляется);
    Google Vision + Gemini без OpenAI -> оба;
- split выполняется по истинному wine_id;
- синтетика добавляется только в train до пропорции ≈40% real / 60% synthetic;
  validation/test остаются полностью реальными.

Выход по умолчанию:
    C:\\dev\\Vino2026\\models\\data_xgboost_synthetic_40_60
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import (  # noqa: E402
    Candidate,
    extract_eval_flags,
    extract_ocr_variant,
    extract_siglip_candidates,
    load_wines,
    resolve_database_url,
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
DEFAULT_OUT = MODELS_DIR / "data_xgboost_synthetic_40_60"

ENGINE_ORDER = ("google_vision", "openai", "gemini")
ENGINE_CODE = {"google_vision": 1, "openai": 2, "gemini": 3}

CYR_TO_LAT = str.maketrans(
    {
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
        "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
        "а": "a", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o",
        "р": "p", "с": "c", "т": "t", "у": "y", "х": "x",
    }
)
LAT_TO_CYR = str.maketrans(
    {
        "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н",
        "O": "О", "P": "Р", "C": "С", "T": "Т", "Y": "У", "X": "Х",
        "a": "а", "e": "е", "k": "к", "m": "м", "h": "н", "o": "о",
        "p": "р", "c": "с", "t": "т", "y": "у", "x": "х",
    }
)

LOW_INFO = {
    "вино", "wine", "виноградное", "защищенного", "географического",
    "указания", "наименования", "места", "происхождения", "россия",
    "russia", "product", "produced", "bottled", "содержит", "contains",
    "год", "года", "объем", "объём", "volume", "алк", "alc", "об",
}
COLOR_TYPE = {
    "красное", "красный", "белое", "белый", "розовое", "розовый",
    "сухое", "сухой", "полусухое", "полусладкое", "сладкое", "игристое",
    "red", "white", "rose", "rosé", "rouge", "blanc", "rosso", "bianco",
    "dry", "sweet", "brut", "брют",
}
YEAR_RE = re.compile(r"(?<!\d)(?:19[5-9]\d|20[0-4]\d)(?!\d)")
VOLUME_ALC_RE = re.compile(
    r"(?:\d+(?:[.,]\d+)?\s*(?:%|мл|ml|л\b|l\b)|(?:алк|alc|vol)\\.?\\s*\\d)",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


@dataclass
class DbScan:
    scan_id: int
    sha256: str
    created_at: datetime | None
    status: dict[str, Any]
    manual_wine_id: int | None
    ocr_variants: list[tuple[str, str]]
    candidates: list[Candidate]
    matched_wine_id: int | None
    false_positive: int
    false_negative: int
    truth_wine_id: int | None
    fp_wrong_wine_id: int | None


@dataclass
class RealGroup:
    sample_id: int
    original_scan_id: int
    sha256: str
    split_group_wine_id: int
    truth_wine_id: int | None
    fp_wrong_wine_id: int | None
    ocr_engine: str
    ocr_text: str
    candidates: list[Candidate]
    split: str | None = None


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def choose_ocr_variants(status: dict[str, Any]) -> list[tuple[str, str]]:
    """Применить пользовательское правило выбора OCR и убрать точные дубли."""
    variants = ((status.get("steps") or {}).get("ocr") or {}).get("variants") or {}
    available: dict[str, str] = {}
    for engine in ENGINE_ORDER:
        lines = extract_ocr_variant(variants.get(engine))
        if lines:
            available[engine] = "\n".join(lines)

    selected: list[tuple[str, str]] = []
    if "google_vision" in available:
        selected.append(("google_vision", available["google_vision"]))
    if "openai" in available:
        selected.append(("openai", available["openai"]))
    elif "gemini" in available:
        selected.append(("gemini", available["gemini"]))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for engine, ocr in selected:
        key = normalize_text(ocr)
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        unique.append((engine, ocr))
    return unique


def load_db_scans(engine, limit: int | None = None) -> list[DbScan]:
    sql = (
        "select id, sha256, created_at, status, manual_wines_id "
        "from search_photos order by id"
    )
    if limit:
        sql += f" limit {int(limit)}"
    with engine.connect() as connection:
        rows = connection.execute(text(sql)).fetchall()

    parsed: list[DbScan] = []
    for row in rows:
        status = row[3] or {}
        ocr_variants = choose_ocr_variants(status)
        candidates = extract_siglip_candidates(status)
        if not ocr_variants or not candidates:
            continue
        manual = _as_int(row[4])
        matched = _as_int(status.get("matched_wine_id"))
        fp, fn = extract_eval_flags(status)

        # Ручная разметка всегда главнее автоматического результата и FP/FN.
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


def deduplicate_scans(scans: list[DbScan]) -> tuple[list[DbScan], int]:
    """Одно фото — одна запись: manual > FP > обычная, затем самая новая."""
    by_sha: dict[str, list[DbScan]] = defaultdict(list)
    for scan in scans:
        by_sha[scan.sha256 or f"id:{scan.scan_id}"].append(scan)
    kept: list[DbScan] = []
    for group in by_sha.values():
        group.sort(
            key=lambda s: (
                int(s.manual_wine_id is not None),
                int(bool(s.false_positive)),
                int(s.truth_wine_id is not None),
                s.scan_id,
            )
        )
        kept.append(group[-1])
    kept.sort(key=lambda s: s.scan_id)
    return kept, len(scans) - len(kept)


def expand_real_groups(scans: list[DbScan]) -> list[RealGroup]:
    groups: list[RealGroup] = []
    sample_id = 1
    for scan in scans:
        split_wine = scan.truth_wine_id or scan.fp_wrong_wine_id
        if split_wine is None:
            continue
        for engine, ocr in scan.ocr_variants:
            groups.append(
                RealGroup(
                    sample_id=sample_id,
                    original_scan_id=scan.scan_id,
                    sha256=scan.sha256,
                    split_group_wine_id=split_wine,
                    truth_wine_id=scan.truth_wine_id,
                    fp_wrong_wine_id=scan.fp_wrong_wine_id,
                    ocr_engine=engine,
                    ocr_text=ocr,
                    candidates=scan.candidates,
                )
            )
            sample_id += 1
    return groups


def assign_split(
    groups: list[RealGroup], train_ratio: float, val_ratio: float, seed: int
) -> None:
    """Split по wine_id с приблизительным балансом количества OCR-групп."""
    by_wine: dict[int, list[RealGroup]] = defaultdict(list)
    for group in groups:
        by_wine[group.split_group_wine_id].append(group)
    wine_ids = list(by_wine)
    random.Random(seed).shuffle(wine_ids)
    total = len(groups)
    seen = 0
    for wine_id in wine_ids:
        fraction = seen / total if total else 0.0
        split = (
            "train"
            if fraction < train_ratio
            else "validation"
            if fraction < train_ratio + val_ratio
            else "test"
        )
        for group in by_wine[wine_id]:
            group.split = split
        seen += len(by_wine[wine_id])


def _words(text_value: str) -> list[re.Match[str]]:
    return list(WORD_RE.finditer(text_value))


def _trim_word(text_value: str, rng: random.Random, *, start: bool) -> tuple[str, str] | None:
    matches = [m for m in _words(text_value) if len(m.group(0)) >= 5]
    if not matches:
        return None
    match = rng.choice(matches)
    word = match.group(0)
    cut = rng.randint(1, min(2, len(word) - 3))
    changed = word[cut:] if start else word[:-cut]
    out = text_value[: match.start()] + changed + text_value[match.end() :]
    return out, f"{'trim_start' if start else 'trim_end'}:{word}->{changed}"


def _swap_homoglyphs(text_value: str, rng: random.Random) -> tuple[str, str] | None:
    positions: list[int] = []
    for i, char in enumerate(text_value):
        if char.translate(CYR_TO_LAT) != char or char.translate(LAT_TO_CYR) != char:
            positions.append(i)
    if not positions:
        return None
    count = min(len(positions), rng.randint(1, 3))
    chosen = set(rng.sample(positions, count))
    chars = list(text_value)
    for pos in chosen:
        char = chars[pos]
        lat = char.translate(CYR_TO_LAT)
        chars[pos] = lat if lat != char else char.translate(LAT_TO_CYR)
    return "".join(chars), f"homoglyphs:{count}"


def _drop_low_info(text_value: str, rng: random.Random) -> tuple[str, str] | None:
    lines = [line.strip() for line in text_value.splitlines() if line.strip()]
    candidates: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        toks = set(re.findall(r"[^\W_]+", normalize_text(line)))
        if toks & LOW_INFO or len(line) <= 5 or VOLUME_ALC_RE.search(line):
            candidates.append((i, line))
    if not candidates or len(lines) <= 2:
        return None
    idx, removed = rng.choice(candidates)
    del lines[idx]
    return "\n".join(lines), f"drop_low_info:{removed}"


def _field_tokens(value: str | None) -> set[str]:
    return {
        token
        for token in re.findall(r"[^\W_]+", normalize_text(value))
        if len(token) >= 4 and token not in LOW_INFO
    }


def _drop_characteristic(
    text_value: str, ref: WineRef, rng: random.Random
) -> tuple[str, str] | None:
    """Удалить одну из характеристик: год, цвет/тип или сорт/регион."""
    options: list[tuple[str, set[str], re.Pattern[str] | None]] = []
    if YEAR_RE.search(text_value):
        options.append(("year", set(), YEAR_RE))
    color_tokens = set(re.findall(r"[^\W_]+", normalize_text(text_value))) & COLOR_TYPE
    if color_tokens:
        options.append(("color_type", color_tokens, None))
    grape_tokens = _field_tokens(ref.grape)
    if grape_tokens:
        options.append(("grape", grape_tokens, None))
    region_tokens = _field_tokens(ref.region)
    if region_tokens:
        options.append(("region", region_tokens, None))
    if not options:
        return None
    name, tokens_to_drop, pattern = rng.choice(options)
    if pattern is not None:
        changed = pattern.sub("", text_value)
    else:
        changed = text_value
        for token in sorted(tokens_to_drop, key=len, reverse=True):
            changed = re.sub(
                rf"(?i)(?<!\w){re.escape(token)}(?!\w)", "", changed
            )
    changed = "\n".join(
        re.sub(r"\s{2,}", " ", line).strip()
        for line in changed.splitlines()
        if re.sub(r"\s{2,}", " ", line).strip()
    )
    if normalize_text(changed) == normalize_text(text_value):
        return None
    return changed, f"drop_characteristic:{name}"


def _drop_random_minor_words(
    text_value: str, ref: WineRef, rng: random.Random
) -> tuple[str, str] | None:
    protected = _field_tokens(ref.name) | _field_tokens(ref.winery)
    matches = [
        m
        for m in _words(text_value)
        if normalize_text(m.group(0)) not in protected
        and len(m.group(0)) <= 8
        and not YEAR_RE.fullmatch(m.group(0))
    ]
    if not matches:
        return None
    count = min(len(matches), rng.randint(1, 2))
    chosen = sorted(rng.sample(matches, count), key=lambda m: m.start(), reverse=True)
    changed = text_value
    removed: list[str] = []
    for match in chosen:
        removed.append(match.group(0))
        changed = changed[: match.start()] + changed[match.end() :]
    changed = "\n".join(
        re.sub(r"\s{2,}", " ", line).strip()
        for line in changed.splitlines()
        if re.sub(r"\s{2,}", " ", line).strip()
    )
    return changed, f"drop_minor:{','.join(removed)}"


def corrupt_ocr(
    source_text: str,
    ref: WineRef,
    seed: int,
    variant_index: int,
    used: set[str],
) -> tuple[str, list[str], str]:
    """Уникальная детерминированная OCR-порча; возвращает text, operations, severity."""
    source_key = normalize_text(source_text)
    for attempt in range(100):
        rng = random.Random(seed * 1_000_003 + variant_index * 10_007 + attempt)
        text_value = source_text
        operations: list[str] = []
        severe = variant_index % 2 == 1
        target_ops = rng.randint(3, 4) if severe else rng.randint(2, 3)

        ops = [
            lambda t: _trim_word(t, rng, start=True),
            lambda t: _trim_word(t, rng, start=False),
            lambda t: _swap_homoglyphs(t, rng),
            lambda t: _drop_low_info(t, rng),
            lambda t: _drop_characteristic(t, ref, rng),
            lambda t: _drop_random_minor_words(t, ref, rng),
        ]
        rng.shuffle(ops)
        # Для сложного positive обязательно попытаться убрать одну характеристику.
        if severe:
            characteristic = _drop_characteristic(text_value, ref, rng)
            if characteristic:
                text_value, op_name = characteristic
                operations.append(op_name)
        for op in ops:
            if len(operations) >= target_ops:
                break
            result = op(text_value)
            if result is None:
                continue
            candidate, op_name = result
            if normalize_text(candidate) != normalize_text(text_value):
                text_value = candidate
                operations.append(op_name)

        key = normalize_text(text_value)
        alpha_tokens = [t for t in re.findall(r"[^\W\d_]+", key) if len(t) >= 2]
        if (
            key
            and key != source_key
            and key not in used
            and len(key) >= 8
            and len(alpha_tokens) >= 2
            and len(operations) >= 2
        ):
            used.add(key)
            return text_value, operations, "hard" if severe else "medium"
    raise RuntimeError(f"Не удалось создать уникальный synthetic OCR для seed={seed}")


def rows_for_group(
    sample_id: int,
    group: RealGroup,
    wines: dict[int, WineRef],
    *,
    ocr_text: str,
    source: str,
    synthetic_meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs: list[WineRef] = []
    candidates: list[Candidate] = []
    for candidate in group.candidates:
        ref = wines.get(candidate.wine_id)
        if ref is not None:
            refs.append(ref)
            candidates.append(candidate)
    features = group_features(
        ocr_text, refs, [candidate.cosine for candidate in candidates]
    )
    rows: list[dict[str, Any]] = []
    for ref, candidate, feature_values in zip(refs, candidates, features):
        if group.truth_wine_id is None:
            if candidate.wine_id != group.fp_wrong_wine_id:
                continue
            label = 0
            kind = "real_fp_hard"
        else:
            label = int(candidate.wine_id == group.truth_wine_id)
            if source == "synthetic":
                kind = "synthetic_tp_hard" if label else "synthetic_tn_embedding"
            else:
                kind = "real_tp" if label else "real_tn_embedding"
        row: dict[str, Any] = {
            "scan_id": sample_id,
            "wine_id": candidate.wine_id,
            "group_id": group.split_group_wine_id,
            "split": group.split,
            "label": label,
            "label_kind": kind,
            "ocr_engine": (
                f"synthetic_{group.ocr_engine}" if source == "synthetic"
                else group.ocr_engine
            ),
            "data_source": source,
            "original_scan_id": group.original_scan_id,
            "siglip_rank_original": candidate.rank,
        }
        row.update(feature_values)
        rows.append(row)
    audit = {
        "sample_id": sample_id,
        "original_scan_id": group.original_scan_id,
        "split": group.split,
        "source": source,
        "ocr_engine": group.ocr_engine,
        "truth_wine_id": group.truth_wine_id,
        "fp_wrong_wine_id": group.fp_wrong_wine_id,
        "candidate_count": len(candidates),
        "pair_count": len(rows),
        "ocr_text": ocr_text,
    }
    if synthetic_meta:
        audit.update(synthetic_meta)
    return rows, audit


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic-ratio", type=float, default=0.60)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.synthetic_ratio < 1.0:
        raise SystemExit("--synthetic-ratio должен быть между 0 и 1")

    db_url = resolve_database_url(args.database_url, args.env_file)
    engine = create_engine(db_url)
    print("[1/7] Читаю search_photos и применяю правила OCR/manual/FP …")
    scans = load_db_scans(engine, args.limit)
    scans_before_dedup = len(scans)
    scans, dropped = deduplicate_scans(scans)
    groups = expand_real_groups(scans)
    print(
        f"      пригодных записей={scans_before_dedup}, после sha256-dedup={len(scans)} "
        f"(удалено {dropped}), реальных OCR-вариантов={len(groups)}"
    )

    print("[2/7] Split по истинному wine_id …")
    assign_split(groups, args.train, args.validation, args.seed)
    print("      ", dict(Counter(group.split for group in groups)))

    wine_ids: set[int] = set()
    for group in groups:
        wine_ids.update(candidate.wine_id for candidate in group.candidates)
        if group.truth_wine_id is not None:
            wine_ids.add(group.truth_wine_id)
        if group.fp_wrong_wine_id is not None:
            wine_ids.add(group.fp_wrong_wine_id)
    wines = load_wines(engine, wine_ids)
    print(f"[3/7] Загружено эталонов вин: {len(wines)}")

    print("[4/7] Строю реальные пары и признаки …")
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit_rows: list[dict[str, Any]] = []
    positive_train_groups: list[RealGroup] = []
    for group in groups:
        rows, audit = rows_for_group(
            group.sample_id, group, wines, ocr_text=group.ocr_text, source="real"
        )
        if not rows:
            continue
        rows_by_split[group.split or "train"].extend(rows)
        audit_rows.append(audit)
        if group.split == "train" and group.truth_wine_id is not None:
            positive_train_groups.append(group)

    real_train_rows = len(rows_by_split["train"])
    target_synth_rows = round(
        real_train_rows * args.synthetic_ratio / (1.0 - args.synthetic_ratio)
    )
    print(
        f"      real train pairs={real_train_rows}; цель synthetic={target_synth_rows} "
        f"({args.synthetic_ratio:.0%} от train)"
    )

    print("[5/7] Генерирую уникальные synthetic OCR и пары …")
    used_texts = {
        normalize_text(group.ocr_text)
        for group in groups
        if normalize_text(group.ocr_text)
    }
    rng = random.Random(args.seed)
    base_order = positive_train_groups[:]
    rng.shuffle(base_order)
    synthetic_rows: list[dict[str, Any]] = []
    synth_audit: list[dict[str, Any]] = []
    variants_per_original: Counter[int] = Counter()
    generation_skipped: Counter[str] = Counter()
    exhausted_originals: set[int] = set()
    next_sample_id = max((group.sample_id for group in groups), default=0) + 1
    cycle = 0
    while base_order and len(synthetic_rows) < target_synth_rows:
        base = base_order[cycle % len(base_order)]
        variant_index = variants_per_original[base.sample_id]
        # Не плодим более четырёх вариантов одного OCR даже при малом наборе.
        if variant_index >= 4 or base.sample_id in exhausted_originals:
            cycle += 1
            if all(
                variants_per_original[g.sample_id] >= 4
                or g.sample_id in exhausted_originals
                for g in base_order
            ):
                break
            continue
        ref = wines.get(base.truth_wine_id or -1)
        if ref is None:
            cycle += 1
            continue
        try:
            synthetic_text, operations, severity = corrupt_ocr(
                base.ocr_text,
                ref,
                seed=args.seed + base.sample_id,
                variant_index=variant_index,
                used=used_texts,
            )
        except RuntimeError:
            # Очень короткий/цифровой OCR невозможно безопасно испортить двумя
            # разными способами, не превратив его в бессмысленный шум.
            exhausted_originals.add(base.sample_id)
            generation_skipped["not_enough_text"] += 1
            cycle += 1
            continue
        rows, audit = rows_for_group(
            next_sample_id,
            base,
            wines,
            ocr_text=synthetic_text,
            source="synthetic",
            synthetic_meta={
                "original_ocr_text": base.ocr_text,
                "variant_index": variant_index,
                "severity": severity,
                "operations": operations,
            },
        )
        if rows:
            synthetic_rows.extend(rows)
            synth_audit.append(audit)
            variants_per_original[base.sample_id] += 1
            next_sample_id += 1
        cycle += 1
    rows_by_split["train"].extend(synthetic_rows)

    out = args.out_dir
    features_dir = out / "features"
    raw_dir = out / "raw"
    features_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[6/7] Записываю отдельный набор …")
    id_cols = [
        "scan_id", "wine_id", "group_id", "split", "label", "label_kind",
        "ocr_engine", "data_source", "original_scan_id", "siglip_rank_original",
    ]
    split_stats: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        rows = rows_by_split[split]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame[id_cols + FEATURE_NAMES]
        frame.to_parquet(features_dir / f"{split}.parquet", index=False)
        sources = Counter(row["data_source"] for row in rows)
        split_stats[split] = {
            "pairs": len(rows),
            "real_pairs": sources.get("real", 0),
            "synthetic_pairs": sources.get("synthetic", 0),
            "synthetic_fraction": (
                sources.get("synthetic", 0) / len(rows) if rows else 0.0
            ),
            "positives": sum(int(row["label"]) for row in rows),
            "negatives": sum(1 - int(row["label"]) for row in rows),
            "label_kinds": dict(Counter(row["label_kind"] for row in rows)),
            "groups": len({row["scan_id"] for row in rows}),
        }

    write_jsonl(raw_dir / "real_groups.jsonl", audit_rows)
    write_jsonl(raw_dir / "synthetic_variants.jsonl", synth_audit)
    stats = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "feature_version": FEATURE_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "requested_train_mix": {
            "real": 1.0 - args.synthetic_ratio,
            "synthetic": args.synthetic_ratio,
        },
        "policy": {
            "positive_priority": [
                "search_photos.manual_wines_id",
                "status.matched_wine_id",
            ],
            "false_positive_without_manual": "matched wine only -> hard negative",
            "false_negative": "not a class; manual overrides; otherwise no special use",
            "ocr": "google_vision + openai; gemini only when openai absent",
            "negatives": "other SigLIP2 Top-20 candidates",
            "synthetic_scope": "train only; validation/test real only",
        },
        "db_scans_eligible_before_dedup": scans_before_dedup,
        "db_scans_after_dedup": len(scans),
        "duplicates_removed": dropped,
        "real_ocr_groups": len(groups),
        "real_group_engines": dict(Counter(g.ocr_engine for g in groups)),
        "manual_truth_groups": sum(
            len(scan.ocr_variants) for scan in scans if scan.manual_wine_id is not None
        ),
        "automatic_truth_groups": sum(
            len(scan.ocr_variants)
            for scan in scans
            if scan.manual_wine_id is None and scan.truth_wine_id is not None
        ),
        "false_positive_negative_only_groups": sum(
            len(scan.ocr_variants)
            for scan in scans
            if scan.truth_wine_id is None and scan.fp_wrong_wine_id is not None
        ),
        "synthetic_variants": len(synth_audit),
        "synthetic_severity": dict(
            Counter(row["severity"] for row in synth_audit)
        ),
        "synthetic_operations": dict(
            Counter(
                op.split(":", 1)[0]
                for row in synth_audit
                for op in row["operations"]
            )
        ),
        "synthetic_generation_skipped": dict(generation_skipped),
        "variants_per_real_train_ocr": {
            str(count): sum(1 for value in variants_per_original.values() if value == count)
            for count in sorted(set(variants_per_original.values()))
        },
        "splits": split_stats,
    }
    (out / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[7/7] Готово")
    for split, split_stat in split_stats.items():
        print(
            f"      {split:<10} pairs={split_stat['pairs']:<7} "
            f"real={split_stat['real_pairs']:<7} synth={split_stat['synthetic_pairs']:<7} "
            f"synth%={split_stat['synthetic_fraction']:.2%} "
            f"pos={split_stat['positives']:<5} neg={split_stat['negatives']}"
        )
    print(
        f"      synthetic OCR-вариантов={len(synth_audit)}, "
        f"папка={out}"
    )


if __name__ == "__main__":
    main()
