"""Похожие / аналогичные вина для сайта.

Без match — критерии из OCR искомого; с match — те же веса по полям
найденного каталожного вина (исключая сам winner). Category/type — каноны
exclusive_lexicon.json (rouge == red). Каталог в памяти при старте backend.

Если по критериям 0 хитов — fallback: top-5 кандидатов embedding с cos>0.7
без hard reject, по убыванию cosine.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Wine
from app.media import label_url
from app.pipeline.exclusive_lexicon import (
    _CATEGORY_GROUPS,
    _LEXICONS,
    _TYPE_GROUPS,
    _compile_form,
    _console_print,
    _cupage_from_label_ocr,
    _dedupe_repeated_words,
    _split_grape_tokens,
    find_lexicon_canons,
)
from app.pipeline.ocr_match import _is_bad_producer_name, normalize_ocr_text

logger = logging.getLogger("vino.analogs")

# Недостающие в OCR критерии не входят в знаменатель (score = Σw·s / Σw_found).
WEIGHTS: dict[str, float] = {
    "winery": 0.35,
    "grape": 0.30,
    "category": 0.20,
    "type": 0.15,
}
# Известный, но другой цвет (красное вместо белого) — штраф поверх нуля по критерию.
COLOR_CONFLICT_FACTOR = 0.75
TOP_N = 10
# Fallback, если по тексту этикетки / критериям нет ни одного аналога:
# top-K кандидатов embedding с cos > порога, без hard reject.
COSINE_FALLBACK_TOP_N = 5
COSINE_FALLBACK_MIN = 0.70

CATEGORY_LABELS = {
    "white": "белое",
    "red": "красное",
    "rose": "розовое",
    "orange": "оранжевое",
}
TYPE_LABELS = {
    "dry": "сухое",
    "semi_dry": "полусухое",
    "semi_sweet": "полусладкое",
    "sweet": "сладкое",
    "brut": "брют",
    "extra_dry": "экстра драй",
}
_TYPE_NEAR_SCORE = 0.4
# У ~половины каталога тип не указан — это не то же самое, что другой тип.
_TYPE_UNKNOWN_SCORE = 0.5
# Каноны без значения в справочнике — в конец сортировки.
_LEX_RANK_MISSING = 10_000

_SEMI_EXTRA_RE = re.compile(r"(?i)\b(semi|extra)[\s\-]+(?=[a-z])")


def _canon_rank(canon: str | None, groups: list[tuple[str, tuple[str, ...]]]) -> int:
    """Индекс канона в exclusive_lexicon (порядок JSON). Нет канона → в конец."""
    if not canon:
        return _LEX_RANK_MISSING
    for i, (c, _) in enumerate(groups):
        if c == canon:
            return i
    return _LEX_RANK_MISSING - 1


def _type_ordinal(canon: str | None) -> int:
    """Порядок сахаристости = порядок type в exclusive_lexicon.json."""
    return _canon_rank(canon, _TYPE_GROUPS)

@dataclass(slots=True, frozen=True)
class _CatalogWine:
    id: int
    name: str | None
    slug: str | None
    photo_name: str | None
    winery: str | None
    winery_canon: str | None
    category: str | None
    color_canon: str | None
    type_canon: str | None
    grapes_display: tuple[str, ...]
    grapes: frozenset[str]


_CATALOG: list[_CatalogWine] = []
_GRAPE_ALIAS: dict[str, str] = {}
_GRAPE_DISPLAY: dict[str, str] = {}
_WINERY_RES: list[tuple[str, str, re.Pattern[str]]] = []
_WINERY_DISPLAY: dict[str, str] = {}
_LOAD_INFO: dict[str, Any] = {}


@lru_cache(maxsize=50_000)
def _norm_cached(text: str) -> str:
    return normalize_ocr_text(text).strip()


def _norm(text: Any) -> str:
    return _norm_cached(str(text or ""))


@lru_cache(maxsize=10_000)
def _dedupe_words(form: str) -> str:
    return _dedupe_repeated_words(form)


def _type_canon(text: str) -> str | None:
    """Первый канон в порядке JSON (semi_* раньше sweet/dry)."""
    if not text:
        return None
    hits = find_lexicon_canons(_SEMI_EXTRA_RE.sub(r"\1", text), "type")
    return next(iter(hits), None)


def _color_canons(text: str) -> list[str]:
    if not text:
        return []
    return list(find_lexicon_canons(text, "category"))


@lru_cache(maxsize=1)
def _color_type_forms() -> frozenset[str]:
    return frozenset(
        _norm(form)
        for lex in ("category", "type")
        for _, forms in _LEXICONS.get(lex) or []
        for form in forms
    )


def _grape_key(token: str) -> str:
    n = _norm(token)
    return _GRAPE_ALIAS.get(n, n)


def _token_prefix(short: str, long: str) -> bool:
    a, b = short.split(), long.split()
    return len(a) < len(b) and b[: len(a)] == a and len(a[0]) >= 5


def _grapes_match(a: str, b: str) -> bool:
    """«рислинг» ~ «рислинг рейнский»; иначе точное равенство ключей."""
    return a == b or _token_prefix(a, b) or _token_prefix(b, a)


def _majority_alias(
    votes: dict[str, Counter[str]], *, min_count: int, min_share: float
) -> dict[str, str]:
    out: dict[str, str] = {}
    for alias, cnt in votes.items():
        target, n = cnt.most_common(1)[0]
        if n >= min_count and n / sum(cnt.values()) >= min_share:
            out[alias] = target
    return out


def load_analog_catalog(db: Session) -> int:
    """Load catalog into memory + Latin→Cyrillic aliases learned from label_ocr."""
    global _CATALOG, _GRAPE_ALIAS, _GRAPE_DISPLAY, _WINERY_RES, _WINERY_DISPLAY
    t0 = time.perf_counter()
    _color_type_forms.cache_clear()
    rows = db.execute(
        select(
            Wine.id,
            Wine.name,
            Wine.slug,
            Wine.photo_name,
            Wine.category,
            Wine.color,
            Wine.region,
            Wine.grape_variety,
            Wine.winery,
            Wine.label,
            Wine.label_ocr,
        )
    ).all()

    # --- grape aliases: cupage (часто латиница) → grape_variety (кириллица)
    grape_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        gv = [_norm(t) for t in _split_grape_tokens(r.grape_variety)]
        cup = [_norm(t) for t in _split_grape_tokens(_cupage_from_label_ocr(r.label_ocr))]
        if len(gv) == 1 and len(cup) == 1 and gv[0] and cup[0] and gv[0] != cup[0]:
            grape_votes[cup[0]][gv[0]] += 1
    grape_alias = _majority_alias(grape_votes, min_count=2, min_share=0.6)

    # --- winery canons + aliases from label_ocr.producer
    winery_display: dict[str, str] = {}
    regions: set[str] = set()
    producer_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        if r.region:
            regions.add(_norm(r.region))
        form = _dedupe_words(str(r.winery or "").strip())
        if not form or _is_bad_producer_name(form):
            continue
        canon = _norm(form)
        if len(canon) < 4:
            continue
        winery_display.setdefault(canon, form)
        lo = r.label_ocr if isinstance(r.label_ocr, dict) else {}
        prod = _dedupe_words(str(lo.get("producer") or "").strip())
        pn = _norm(prod)
        if pn and pn != canon and len(pn) >= 4 and not _is_bad_producer_name(prod):
            producer_votes[pn][canon] += 1
    winery_alias = {
        alias: canon
        for alias, canon in _majority_alias(
            producer_votes, min_count=1, min_share=0.8
        ).items()
        if alias not in winery_display
        and alias not in regions
        and not find_lexicon_canons(alias, "category")
        and not find_lexicon_canons(alias, "type")
    }
    forms: list[tuple[str, str]] = [(c, c) for c in winery_display]
    forms += [(canon, alias) for alias, canon in winery_alias.items()]
    forms.sort(key=lambda cf: (-len(cf[1]), cf[1]))
    winery_res = [(canon, form, _compile_form(form)) for canon, form in forms]

    _GRAPE_ALIAS = grape_alias
    grape_display: dict[str, str] = {}
    catalog: list[_CatalogWine] = []
    for r in rows:
        gv_tokens = _split_grape_tokens(r.grape_variety)
        cup_tokens = _split_grape_tokens(_cupage_from_label_ocr(r.label_ocr))
        grapes: set[str] = set()
        for t in gv_tokens + cup_tokens:
            k = _grape_key(t)
            if len(k) >= 3:
                grapes.add(k)
        for t in gv_tokens:
            grape_display.setdefault(_grape_key(t), t.strip())
        lo = r.label_ocr if isinstance(r.label_ocr, dict) else {}
        colors = (
            _color_canons(str(r.category or ""))
            or _color_canons(str(lo.get("color") or ""))
            or _color_canons(str(r.name or ""))
        )
        typ = (
            _type_canon(str(lo.get("type") or ""))
            or _type_canon(str(r.name or ""))
            or _type_canon(str(r.label or ""))
        )
        w_form = _dedupe_words(str(r.winery or "").strip())
        w_canon = _norm(w_form) or None
        catalog.append(
            _CatalogWine(
                id=int(r.id),
                name=r.name,
                slug=r.slug,
                photo_name=r.photo_name,
                winery=r.winery,
                winery_canon=w_canon,
                category=r.category,
                color_canon=colors[0] if colors else None,
                type_canon=typ,
                grapes_display=tuple(t.strip() for t in (gv_tokens or cup_tokens)),
                grapes=frozenset(grapes),
            )
        )

    _CATALOG = catalog
    _GRAPE_DISPLAY = grape_display
    _WINERY_RES = winery_res
    _WINERY_DISPLAY = winery_display
    ms = round((time.perf_counter() - t0) * 1000, 1)
    _LOAD_INFO.clear()
    _LOAD_INFO.update(
        {
            "wines": len(catalog),
            "grape_aliases": len(grape_alias),
            "winery_aliases": len(winery_alias),
            "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ms": ms,
        }
    )
    msg = (
        f"analog catalog loaded: wines={len(catalog)} "
        f"grape_aliases={len(grape_alias)} winery_aliases={len(winery_alias)} "
        f"({ms} ms)"
    )
    logger.info(msg)
    _console_print(f"[analogs] {msg}")
    return len(catalog)


def extract_analog_criteria(query_text: str) -> dict[str, Any]:
    """Критерии из OCR искомого (каноны)."""
    text = query_text or ""
    norm = normalize_ocr_text(text)
    hay = f"{text}\n{norm}"

    wineries: list[tuple[str, str]] = []
    for canon, form, cre in _WINERY_RES:
        if any(canon == c for c, _ in wineries):
            continue
        # короче уже найденной формы и внутри неё — та же надпись
        if any(_norm(form) in _norm(f) for _, f in wineries):
            continue
        if cre.search(hay):
            wineries.append((canon, form))
        if len(wineries) >= 3:
            break

    raw_grapes = list(find_lexicon_canons(text, "grape"))
    grape_keys: list[str] = []
    for canon in raw_grapes:
        if canon in _color_type_forms():
            continue
        # «sauvignon» внутри «cabernet sauvignon» — не отдельный сорт
        if any(
            o != canon and re.search(rf"(?<!\w){re.escape(canon)}(?!\w)", o)
            for o in raw_grapes
        ):
            continue
        k = _GRAPE_ALIAS.get(canon, canon)
        if k not in grape_keys:
            grape_keys.append(k)
    grape_keys = [
        k for k in grape_keys if not any(_token_prefix(k, o) for o in grape_keys)
    ]

    return {
        "winery": [c for c, _ in wineries],
        "category": _color_canons(text),
        "type": _type_canon(text),
        "grape": grape_keys,
    }


def _criteria_public(crit: dict[str, Any]) -> dict[str, Any]:
    return {
        "winery": [_WINERY_DISPLAY.get(c, c) for c in crit["winery"]],
        "category": [CATEGORY_LABELS.get(c, c) for c in crit["category"]],
        "category_canons": list(crit["category"]),
        "type": TYPE_LABELS.get(crit["type"], crit["type"]) if crit["type"] else None,
        "type_canon": crit["type"],
        "grape": [_GRAPE_DISPLAY.get(k, k) for k in crit["grape"]],
    }


def _catalog_by_id() -> dict[int, _CatalogWine]:
    return {w.id: w for w in _CATALOG}


def _item_from_catalog(
    w: _CatalogWine,
    *,
    score: float,
    cosine: float | None,
    matched: dict[str, Any] | None = None,
    matched_count: int = 0,
    criteria_count: int = 0,
    grape_hits: list[str] | None = None,
) -> dict[str, Any]:
    hits = grape_hits or []
    m = matched or {}
    return {
        "id": w.id,
        "name": w.name,
        "slug": w.slug,
        "winery": w.winery,
        "category": CATEGORY_LABELS.get(w.color_canon or "", w.category),
        "type": TYPE_LABELS.get(w.type_canon or "") if w.type_canon else None,
        "grapes": [
            {
                "name": g,
                "matched": any(_grapes_match(q, _grape_key(g)) for q in hits),
            }
            for g in w.grapes_display
        ],
        "matched": m,
        "matched_count": matched_count,
        "criteria_count": criteria_count,
        "score": round(score, 4),
        "cosine": round(cosine, 4) if cosine is not None and cosine >= 0 else None,
        "label_url": label_url(w.photo_name, slug=w.slug),
    }


def _items_by_embedding_cosine(
    cosine_by_id: dict[int, float],
    *,
    top_n: int = COSINE_FALLBACK_TOP_N,
    min_cos: float = COSINE_FALLBACK_MIN,
    exclude_ids: set[int] | frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    """Top-N среди кандидатов embedding: cos > min_cos, без exclude/hard-reject."""
    by_id = _catalog_by_id()
    skip = {int(x) for x in (exclude_ids or set())}
    ranked: list[tuple[int, float]] = []
    thr = float(min_cos)
    for wid_raw, cos_raw in (cosine_by_id or {}).items():
        try:
            wid = int(wid_raw)
            cos = float(cos_raw)
        except (TypeError, ValueError):
            continue
        if wid in skip or wid not in by_id:
            continue
        if cos <= thr:
            continue
        ranked.append((wid, cos))
    ranked.sort(key=lambda t: (-t[1], t[0]))
    items: list[dict[str, Any]] = []
    for wid, cos in ranked[:top_n]:
        items.append(
            _item_from_catalog(
                by_id[wid],
                score=cos,
                cosine=cos,
                matched={},
                matched_count=0,
                criteria_count=0,
            )
        )
    return items


def _rank_analogs(
    crit: dict[str, Any],
    *,
    cosine_by_id: dict[int, float] | None = None,
    top_n: int = TOP_N,
    exclude_ids: set[int] | None = None,
    hard_reject_ids: set[int] | frozenset[int] | None = None,
    source: str = "criteria",
    seed_wine_id: int | None = None,
) -> dict[str, Any]:
    """Rank catalog wines by OCR/catalog criteria; cosine fallback if empty."""
    t0 = time.perf_counter()
    cos = cosine_by_id or {}
    skip = set(exclude_ids or set())
    hard = {int(x) for x in (hard_reject_ids or set())}
    found = [k for k in WEIGHTS if crit.get(k)]
    available = sum(WEIGHTS[k] for k in found)
    base: dict[str, Any] = {
        "ok": True,
        "criteria": _criteria_public(crit),
        "criteria_found": found,
        "weights": dict(WEIGHTS),
        "catalog": dict(_LOAD_INFO),
        "cosine_fallback_min": COSINE_FALLBACK_MIN,
        "cosine_fallback_top_n": COSINE_FALLBACK_TOP_N,
    }
    if seed_wine_id is not None:
        base["seed_wine_id"] = int(seed_wine_id)
    if not _CATALOG:
        base.update(ok=False, error="catalog not loaded", items=[])
        base["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return base

    def _cosine_fallback(reason: str) -> dict[str, Any]:
        items = _items_by_embedding_cosine(
            cos,
            top_n=COSINE_FALLBACK_TOP_N,
            min_cos=COSINE_FALLBACK_MIN,
            exclude_ids=skip | hard,
        )
        base.update(
            items=items,
            pool=len(items),
            reason=reason,
            source="embedding_cosine",
            skipped=False,
            hard_reject_excluded=len(hard),
        )
        if not items:
            base["skipped"] = True
            base["note"] = (
                f"no candidates with cos>{COSINE_FALLBACK_MIN:.2f} "
                "after hard-reject filter"
            )
        base["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return base

    if not found:
        return _cosine_fallback("no_criteria_cosine_fallback")

    q_wineries = set(crit["winery"])
    q_colors = set(crit["category"])
    q_type = crit["type"]
    q_grapes: list[str] = list(crit["grape"] or [])

    scored: list[tuple[float, float, int, _CatalogWine, dict[str, Any]]] = []
    for w in _CATALOG:
        if w.id in skip:
            continue
        raw = 0.0
        n_hit = 0
        conflict = False
        m: dict[str, Any] = {}
        if q_wineries:
            ok = w.winery_canon in q_wineries
            m["winery"] = ok
            if ok:
                raw += WEIGHTS["winery"]
                n_hit += 1
        if q_grapes:
            hits = [q for q in q_grapes if any(_grapes_match(q, g) for g in w.grapes)]
            if hits:
                recall = len(hits) / len(q_grapes)
                share = min(1.0, len(hits) / max(1, len(w.grapes_display)))
                raw += WEIGHTS["grape"] * (0.75 * recall + 0.25 * share)
                n_hit += 1
            m["grape"] = bool(hits)
            m["grape_hits"] = hits
        if q_colors:
            ok = w.color_canon in q_colors
            m["category"] = ok
            if ok:
                raw += WEIGHTS["category"]
                n_hit += 1
            elif w.color_canon:
                conflict = True
        if q_type:
            if w.type_canon == q_type:
                m["type"] = True
                raw += WEIGHTS["type"]
                n_hit += 1
            elif w.type_canon is None:
                m["type"] = None
                raw += WEIGHTS["type"] * _TYPE_UNKNOWN_SCORE
            elif abs(_type_ordinal(w.type_canon) - _type_ordinal(q_type)) <= 1:
                m["type"] = "partial"
                raw += WEIGHTS["type"] * _TYPE_NEAR_SCORE
            else:
                m["type"] = False
        if raw <= 0:
            continue
        score = raw / available
        if conflict:
            score *= COLOR_CONFLICT_FACTOR
        scored.append((score, float(cos.get(w.id, -1.0)), n_hit, w, m))

    if not scored:
        return _cosine_fallback("no_match_cosine_fallback")

    # После score: категория и тип в порядке exclusive_lexicon.json, затем cosine.
    scored.sort(
        key=lambda t: (
            -t[0],
            _canon_rank(t[3].color_canon, _CATEGORY_GROUPS),
            _canon_rank(t[3].type_canon, _TYPE_GROUPS),
            -t[1],
            -t[2],
            t[3].id,
        )
    )
    items: list[dict[str, Any]] = []
    for score, cosine, n_hit, w, m in scored[:top_n]:
        grape_hits = m.pop("grape_hits", [])
        items.append(
            _item_from_catalog(
                w,
                score=score,
                cosine=cosine if cosine >= 0 else None,
                matched=m,
                matched_count=n_hit,
                criteria_count=len(found),
                grape_hits=grape_hits,
            )
        )
    base.update(items=items, pool=len(scored), source=source)
    base["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return base


def find_analogs(
    query_text: str,
    *,
    cosine_by_id: dict[int, float] | None = None,
    hard_reject_ids: set[int] | frozenset[int] | None = None,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    """Top-N вин каталога по совпадению критериев OCR искомого.

    Если критерии из текста не извлечены или подходящих вин нет —
    top-5 кандидатов с cos>0.7 без hard reject (по убыванию cos).
    """
    return _rank_analogs(
        extract_analog_criteria(query_text),
        cosine_by_id=cosine_by_id,
        hard_reject_ids=hard_reject_ids,
        top_n=top_n,
        source="criteria",
    )


def find_analogs_from_wine(
    wine_id: int,
    *,
    cosine_by_id: dict[int, float] | None = None,
    hard_reject_ids: set[int] | frozenset[int] | None = None,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    """Похожие вина по полям найденного каталожного вина (без самого winner)."""
    by_id = _catalog_by_id()
    seed = by_id.get(int(wine_id))
    if seed is None:
        return {
            "ok": False,
            "error": f"wine {wine_id} not in analog catalog",
            "items": [],
            "criteria": {
                "winery": [],
                "category": [],
                "type": None,
                "grape": [],
            },
            "criteria_found": [],
            "source": "matched_wine_catalog",
            "seed_wine_id": int(wine_id),
        }
    crit: dict[str, Any] = {
        "winery": [seed.winery_canon] if seed.winery_canon else [],
        "category": [seed.color_canon] if seed.color_canon else [],
        "type": seed.type_canon,
        "grape": sorted(seed.grapes),
    }
    return _rank_analogs(
        crit,
        cosine_by_id=cosine_by_id,
        top_n=top_n,
        exclude_ids={seed.id},
        hard_reject_ids=hard_reject_ids,
        source="matched_wine_catalog",
        seed_wine_id=seed.id,
    )
