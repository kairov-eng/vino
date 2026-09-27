"""Синтетические запросы против FP: twin / brand / garbage / attribute conflict."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from prepare_dataset import Candidate
from prepare_xgboost_synthetic import RealGroup, corrupt_ocr
from text_features import WineRef, normalize_text, tokens

_COLOR_SWAP = [
    ("красное", "белое"),
    ("красный", "белый"),
    ("белое", "красное"),
    ("белый", "красный"),
    ("розовое", "красное"),
    ("розовый", "красный"),
    ("red", "white"),
    ("white", "red"),
    ("rose", "red"),
    ("rosé", "red"),
]
_SWEET_SWAP = [
    ("сухое", "сладкое"),
    ("сухой", "сладкий"),
    ("полусухое", "полусладкое"),
    ("сладкое", "сухое"),
    ("сладкий", "сухой"),
    ("dry", "sweet"),
    ("sweet", "dry"),
    ("semi-dry", "semi-sweet"),
]
_GARBAGE_TEMPLATES = [
    "399\n699\n3995",
    "199 299 499",
    "SKU-88421\n12.990",
    "0,75\n13%\n2024",
    "цена\n890\nруб",
    "399 , 699 399",
    "10020\n750 ml\n13.5%",
]


def _norm_winery(value: str | None) -> str:
    return normalize_text(value or "")


def _name_tokens(ref: WineRef) -> set[str]:
    return {t for t in tokens(normalize_text(ref.name or "")) if len(t) >= 3}


def _label_lines(ref: WineRef) -> list[str]:
    return [ln.strip() for ln in str(ref.label or "").splitlines() if ln.strip()]


def find_twins(
    truth: WineRef,
    candidates: list[Candidate],
    wines: dict[int, WineRef],
) -> list[WineRef]:
    """Близнецы в Top-N: тот же winery или пересечение name ≥ 2 токена."""
    twins: list[WineRef] = []
    tw = _norm_winery(truth.winery)
    tt = _name_tokens(truth)
    for cand in candidates:
        if cand.wine_id == truth.wine_id:
            continue
        other = wines.get(cand.wine_id)
        if other is None:
            continue
        ow = _norm_winery(other.winery)
        ot = _name_tokens(other)
        same_winery = bool(tw and ow and (tw == ow or tw in ow or ow in tw))
        name_overlap = len(tt & ot) >= 2
        if same_winery or name_overlap:
            twins.append(other)
    return twins


def _inject_twin_tokens(base_text: str, twin: WineRef, rng: random.Random) -> str | None:
    """Подмешать уникальные токены близнеца в текст правильного вина."""
    base_toks = set(tokens(normalize_text(base_text)))
    twin_unique = [
        t
        for t in tokens(normalize_text(f"{twin.name or ''} {twin.grape or ''} {twin.label or ''}"))
        if len(t) >= 4 and t not in base_toks
    ]
    if not twin_unique:
        return None
    pick = rng.sample(twin_unique, k=min(2, len(twin_unique)))
    lines = [ln.strip() for ln in base_text.splitlines() if ln.strip()]
    if not lines:
        lines = [normalize_text(base_text)]
    insert_at = rng.randint(0, len(lines))
    lines.insert(insert_at, " ".join(pick).upper())
    # иногда выкинуть уникальную строку эталона (сорт)
    truth_only = [
        i
        for i, ln in enumerate(lines)
        if any(len(t) >= 4 and t not in set(tokens(normalize_text(twin.label or ""))) for t in tokens(normalize_text(ln)))
    ]
    if truth_only and rng.random() < 0.6 and len(lines) > 2:
        del lines[rng.choice(truth_only)]
    out = "\n".join(lines)
    if normalize_text(out) == normalize_text(base_text):
        return None
    return out


def _drop_unique_name_grape(ref: WineRef) -> str | None:
    """Этикетка без уникальных токенов name/grape — общий бренд остаётся."""
    winery_toks = set(tokens(normalize_text(ref.winery or "")))
    drop = (_name_tokens(ref) | {t for t in tokens(normalize_text(ref.grape or "")) if len(t) >= 4}) - winery_toks
    if not drop:
        return None
    lines_out: list[str] = []
    for line in _label_lines(ref):
        kept = [
            w
            for w in re.findall(r"\S+", line)
            if normalize_text(w) not in drop and not any(normalize_text(w) == d for d in drop)
        ]
        # filter tokens that equal dropped
        kept2 = []
        for w in re.findall(r"\S+", line):
            nt = normalize_text(re.sub(r"[^\w]+", "", w, flags=re.UNICODE))
            if nt and nt in drop:
                continue
            kept2.append(w)
        if kept2:
            lines_out.append(" ".join(kept2))
    text = "\n".join(lines_out).strip()
    if len(normalize_text(text)) < 8:
        return None
    if normalize_text(text) == normalize_text(ref.label or ""):
        return None
    return text


def _swap_attribute(text: str, rng: random.Random) -> tuple[str, str] | None:
    pairs = _COLOR_SWAP + _SWEET_SWAP
    rng.shuffle(pairs)
    for src, dst in pairs:
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(src)}(?!\w)")
        if pattern.search(text):
            return pattern.sub(dst, text, count=1), f"{src}->{dst}"
    return None


def _stamp(group: RealGroup, **attrs: Any) -> RealGroup:
    for key, value in attrs.items():
        setattr(group, key, value)
    return group


def expand_fp_synthetics(
    groups: list[RealGroup],
    wines: dict[int, WineRef],
    *,
    seed: int = 42,
    max_twin_per_group: int = 2,
    max_corrupt_per_group: int = 2,
) -> tuple[list[RealGroup], Counter]:
    """Добавить синтетические train-группы A–D (+ лёгкая OCR-порча)."""
    stats: Counter = Counter()
    extra: list[RealGroup] = []
    next_id = max((g.sample_id for g in groups), default=0) + 1
    used_texts: dict[int, set[str]] = defaultdict(set)
    rng = random.Random(seed)

    train_pos = [
        g
        for g in groups
        if g.split == "train"
        and g.truth_wine_id is not None
        and getattr(g, "_query_source", "scan_ocr") == "scan_ocr"
    ]

    for group in train_pos:
        truth = wines.get(group.truth_wine_id or -1)
        if truth is None:
            continue
        base_key = normalize_text(group.ocr_text)
        used_texts[group.original_scan_id].add(base_key)

        # A) twin-swap
        twins = find_twins(truth, group.candidates, wines)
        for twin in twins[:max_twin_per_group]:
            mixed = _inject_twin_tokens(group.ocr_text, twin, random.Random(rng.randint(1, 10**9)))
            if mixed is None:
                mixed = _inject_twin_tokens(str(truth.label or ""), twin, random.Random(rng.randint(1, 10**9)))
            if mixed is None:
                stats["twin_skip"] += 1
                continue
            key = normalize_text(mixed)
            if key in used_texts[group.original_scan_id]:
                stats["twin_dup"] += 1
                continue
            used_texts[group.original_scan_id].add(key)
            g = replace(
                group,
                sample_id=next_id,
                ocr_engine=f"synth_twin_{group.ocr_engine}",
                ocr_text=mixed,
            )
            _stamp(
                g,
                _query_source="synth_twin",
                _base_sample_id=group.sample_id,
                _catalog_sim=None,
                _sample_weight=3.0,
                _all_negative=False,
                _synth_kind=None,
                _data_source="synthetic",
            )
            # twin negative получает ещё больший вес на уровне строки twin wine —
            # помечаем id близнеца для build_rows через _hard_neg_wine_ids
            setattr(g, "_boost_neg_wine_ids", {twin.wine_id})
            extra.append(g)
            next_id += 1
            stats["twin"] += 1

        # B) shared-brand depleted query
        depleted = _drop_unique_name_grape(truth)
        if depleted is not None:
            key = normalize_text(depleted)
            if key not in used_texts[group.original_scan_id]:
                used_texts[group.original_scan_id].add(key)
                g = replace(
                    group,
                    sample_id=next_id,
                    ocr_engine=f"synth_brand_{group.ocr_engine}",
                    ocr_text=depleted,
                )
                _stamp(
                    g,
                    _query_source="synth_brand",
                    _base_sample_id=group.sample_id,
                    _catalog_sim=None,
                    _sample_weight=2.5,
                    _all_negative=False,
                    _synth_kind=None,
                    _data_source="synthetic",
                )
                extra.append(g)
                next_id += 1
                stats["brand"] += 1
            else:
                stats["brand_dup"] += 1
        else:
            stats["brand_skip"] += 1

        # D) attribute conflict → all negatives
        swapped = _swap_attribute(group.ocr_text, random.Random(rng.randint(1, 10**9)))
        if swapped is None:
            swapped = _swap_attribute(str(truth.label or ""), random.Random(rng.randint(1, 10**9)))
        if swapped is not None:
            text_value, op = swapped
            key = normalize_text(text_value)
            if key not in used_texts[group.original_scan_id] and len(key) >= 8:
                used_texts[group.original_scan_id].add(key)
                g = replace(
                    group,
                    sample_id=next_id,
                    ocr_engine=f"synth_attr_{group.ocr_engine}",
                    ocr_text=text_value,
                    truth_wine_id=None,
                    fp_wrong_wine_id=None,
                )
                _stamp(
                    g,
                    _query_source="synth_attr",
                    _base_sample_id=group.sample_id,
                    _catalog_sim=None,
                    _sample_weight=3.0,
                    _all_negative=True,
                    _synth_kind="synth_attr_conflict",
                    _data_source="synthetic",
                    _synth_meta={"op": op},
                )
                extra.append(g)
                next_id += 1
                stats["attr"] += 1
            else:
                stats["attr_dup"] += 1
        else:
            stats["attr_skip"] += 1

        # OCR corruption (усиление positives с дырами)
        for vi in range(max_corrupt_per_group):
            try:
                corrupted, ops, severity = corrupt_ocr(
                    group.ocr_text,
                    truth,
                    seed=group.original_scan_id,
                    variant_index=vi,
                    used=used_texts[group.original_scan_id],
                )
            except RuntimeError:
                stats["corrupt_skip"] += 1
                continue
            g = replace(
                group,
                sample_id=next_id,
                ocr_engine=f"synth_corrupt_{group.ocr_engine}",
                ocr_text=corrupted,
            )
            _stamp(
                g,
                _query_source="synth_corrupt",
                _base_sample_id=group.sample_id,
                _catalog_sim=None,
                _sample_weight=1.5,
                _all_negative=False,
                _synth_kind=None,
                _data_source="synthetic",
                _synth_meta={"operations": ops, "severity": severity},
            )
            extra.append(g)
            next_id += 1
            stats["corrupt"] += 1

    # C) garbage OCR — все negatives на кандидатах train-групп
    garbage_sources = [
        g
        for g in train_pos
        if g.candidates
    ]
    rng.shuffle(garbage_sources)
    n_garbage = min(len(garbage_sources), max(80, len(train_pos) // 3))
    for i, group in enumerate(garbage_sources[:n_garbage]):
        text_value = _GARBAGE_TEMPLATES[i % len(_GARBAGE_TEMPLATES)]
        # чуть рандомизируем числа
        text_value = re.sub(
            r"\d",
            lambda m: str(rng.randint(0, 9)),
            text_value,
            count=rng.randint(1, 4),
        )
        g = replace(
            group,
            sample_id=next_id,
            ocr_engine="synth_garbage",
            ocr_text=text_value,
            truth_wine_id=None,
            fp_wrong_wine_id=None,
        )
        _stamp(
            g,
            _query_source="synth_garbage",
            _base_sample_id=group.sample_id,
            _catalog_sim=None,
            _sample_weight=2.5,
            _all_negative=True,
            _synth_kind="synth_garbage",
            _data_source="synthetic",
        )
        extra.append(g)
        next_id += 1
        stats["garbage"] += 1

    return extra, stats


def force_fp_groups_to_train(groups: list[RealGroup]) -> int:
    """Все FP-группы (truth is None) → train, чтобы не терять редкие ошибки."""
    moved = 0
    for group in groups:
        if group.truth_wine_id is None and group.split != "train":
            group.split = "train"
            moved += 1
    return moved
