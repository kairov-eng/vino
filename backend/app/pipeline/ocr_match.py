"""Multi-stage OCR text matching for wine_id (key fields + weighted score).

Key catalog fields: winery, name, color/category, grape, type (dry/sweet…), vintage.
Hard mismatch on any key except vintage → TextScore=FinalScore=0.
Vintage may differ (same wine, other year) — soft penalty only.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.config import (
    FINAL_SCORE_W_COS,
    FINAL_SCORE_W_TEXT,
    TEXT_WEIGHT_BRAND,
    TEXT_WEIGHT_COLOR,
    TEXT_WEIGHT_GRAPE,
    TEXT_WEIGHT_OTHER,
    TEXT_WEIGHT_PRODUCER,
    TEXT_WEIGHT_REGION,
    TEXT_WEIGHT_TYPE,
    TEXT_WEIGHT_VINTAGE,
    TEXT_WEIGHT_WINE,
)
from app.db.models import Wine
from app.pipeline.ocr_match_v2 import (
    SOFT_CHANNELS,
    SOFT_VETO_COVER_Q,
    SOFT_VETO_SCORE2,
    build_catalog_idf,
    score_channel_soft,
)
from app.pipeline.text_compare import normalize_compare_text

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None  # type: ignore

# Кеш списков виноделен/регионов для extract_entities (по n_docs).
_PRODUCERS_REGIONS_CACHE: dict[str, Any] = {
    "n_docs": -1,
    "producers": [],
    "regions": [],
}

_VINTAGE_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_YEAR_OCR_RE = re.compile(
    r"\b(?:19|20)[\dOoIl]{2}\b",
    re.IGNORECASE,
)

_NOISE_RE = re.compile(
    r"\b("
    r"contains?\s+sulph?ites?|сульфит\w*|алк(?:оголь)?|alcohol|"
    r"\d+[.,]\d+\s*%|\d+\s*%|750\s*ml|0[,.]75\s*л|75\s*cl|"
    r"объ[её]м|volume|produced|изготов\w*|розлив\w*|bottled|"
    r"wine\s+of|table\s+wine|столов\w*\s+вин\w*|mis\s+en\s+bouteille|"
    r"au\s+chateau|выдержан\w*|"
    r"top\s*100\s*wines?|винодельн\w*\s+года|winery\s+of\s+the\s+year"
    r")\b",
    re.IGNORECASE,
)

# «Винодельня 78» / голые «шато» — шум наград / слишком общие ярлыки
_BAD_PRODUCER_RE = re.compile(
    r"^(?:винодельн\w*|winery|chateau|шато|дом)\s*\d*$",
    re.IGNORECASE,
)

_STOP_TOKENS = {
    "вино",
    "wine",
    "wines",
    "top",
    "100",
    "белое",
    "красное",
    "розовое",
    "сухое",
    "полусухое",
    "полусладкое",
    "сладкое",
    "брют",
    "extra",
    "brut",
    "года",
    "год",
    "year",
    "the",
    "and",
    "для",
    "из",
    "или",
    "senoabr",
    "vol",
    "фермерское",
    "сделано",
    "деревне",
    "винной",
    "россии",
    "кубань",
}

# Общие слова в названии винодельни — без уникального токена матч ненадёжен
_GENERIC_PRODUCER_TOKENS = {
    "семейная",
    "семеиная",  # NFKD: й → и + breve
    "семейныи",
    "семеиныи",
    "винодельня",
    "винодельни",
    "винодельныи",
    "winery",
    "wineries",
    "chateau",
    "шато",
    "дом",
    "хозяйство",
    "vineyards",
    "vineyard",
    "estate",
    "agricole",
    "production",
    "premium",
    "премиум",
    "виннои",
    "винныи",
    "деревне",
    "деревня",
}

# Цвет / категория на этикетке → канон
_COLOR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bбел(?:ы[йех]|ое|ая|ым)\b", re.IGNORECASE), "white"),
    (re.compile(r"\bкрасн(?:ы[йех]|ое|ая|ым)\b", re.IGNORECASE), "red"),
    (re.compile(r"\bрозов(?:ы[йех]|ое|ая|ым)\b", re.IGNORECASE), "rose"),
    (re.compile(r"\bоранж\w*\b", re.IGNORECASE), "orange"),
    (re.compile(r"\bчерн(?:ы[йех]|ое|ая|ым)\b", re.IGNORECASE), "black"),
]

_CAT_TO_COLOR = {
    "белое": "white",
    "красное": "red",
    "розовое": "rose",
    "оранжевое": "orange",
}

# Тип (сахаристость) — порядок: сначала составные
_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bполусладк\w*\b", re.IGNORECASE), "semi_sweet"),
    (re.compile(r"\bполусух\w*\b", re.IGNORECASE), "semi_dry"),
    (re.compile(r"\bсладк\w*\b", re.IGNORECASE), "sweet"),
    (re.compile(r"\bсух(?:ое|ой|ая|ие)?\b", re.IGNORECASE), "dry"),
    (re.compile(r"\b(?:extra\s*)?brut\b|\bбрют\b", re.IGNORECASE), "brut"),
]

_COMMON_GRAPES = (
    "мерло",
    "мерла",
    "каберне",
    "совиньон",
    "шардоне",
    "пино",
    "нуар",
    "рислинг",
    "саперави",
    "мускат",
    "мускатель",
    "алиготе",
    "ркацители",
    "сира",
    "шираз",
    "темпранильо",
    "мальбек",
    "совиньон блан",
    "каберне совиньон",
    "красностоп",
    "рубиновый",
    "рубин",
)

# Порог soft-sim (0..100), ниже — конфликт ключа
_KEY_MATCH_MIN = 62.0
_KEY_CONFLICT_MAX = 45.0

# Слабая структура строк OCR↔label + cos ниже порога → точно другое вино
_LINE_STRUCT_COS_MAX = 0.80
_LINE_STRUCT_MATCH_MIN = 65.0  # sim линии, чтобы считать совпавшей
_GENERIC_LABEL_LINES = frozenset(
    {
        "винодельня",
        "винодельни",
        "вино",
        "wine",
        "winery",
        "этикетка",
    }
)


def text_score_weights() -> dict[str, float]:
    try:
        from app.pipeline.runtime_settings import get_text_weights

        base = get_text_weights()
    except Exception:  # noqa: BLE001
        base = {
            "wine": TEXT_WEIGHT_WINE,
            "producer": TEXT_WEIGHT_PRODUCER,
            "color": TEXT_WEIGHT_COLOR,
            "type": TEXT_WEIGHT_TYPE,
            "grape": TEXT_WEIGHT_GRAPE,
            "vintage": TEXT_WEIGHT_VINTAGE,
            "region": TEXT_WEIGHT_REGION,
            "other": TEXT_WEIGHT_OTHER,
            "brand": TEXT_WEIGHT_BRAND,
        }
    out = dict(base)
    out.setdefault("wine", TEXT_WEIGHT_WINE)
    out.setdefault("producer", TEXT_WEIGHT_PRODUCER)
    out.setdefault("color", TEXT_WEIGHT_COLOR)
    out.setdefault("type", TEXT_WEIGHT_TYPE)
    out.setdefault("grape", TEXT_WEIGHT_GRAPE)
    out.setdefault("vintage", TEXT_WEIGHT_VINTAGE)
    out.setdefault("region", TEXT_WEIGHT_REGION)
    out.setdefault("other", TEXT_WEIGHT_OTHER)
    out.setdefault("brand", TEXT_WEIGHT_BRAND)
    return out


def final_score_weights() -> dict[str, float]:
    try:
        from app.pipeline.runtime_settings import get_final_weights

        fw = get_final_weights()
        return {
            "text": float(fw.get("text", FINAL_SCORE_W_TEXT)),
            "cosine": float(fw.get("cosine", FINAL_SCORE_W_COS)),
        }
    except Exception:  # noqa: BLE001
        return {
            "text": float(FINAL_SCORE_W_TEXT),
            "cosine": float(FINAL_SCORE_W_COS),
        }


def normalize_ocr_text(text: str) -> str:
    """Unicode NFKC, strip accents, ё→е, OCR year fixes, drop boilerplate, lowercase."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("\r", "\n").replace("ё", "е").replace("Ё", "е")

    def _fix_year(m: re.Match[str]) -> str:
        s = m.group(0)
        return (
            s.replace("O", "0")
            .replace("o", "0")
            .replace("I", "1")
            .replace("l", "1")
            .replace("L", "1")
        )

    t = _YEAR_OCR_RE.sub(_fix_year, t)
    t = _NOISE_RE.sub(" ", t)
    t = re.sub(r"\b\d+[.,]?\d*\s*%(\s*vol\.?)?\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d+[.,]?\d*\s*vol\.?\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"[^\w\s\-А-Яа-яA-Za-z0-9]+", " ", t, flags=re.UNICODE)
    return normalize_compare_text(t)


def extract_vintage(text: str) -> str | None:
    years = _VINTAGE_RE.findall(normalize_ocr_text(text) if text else "")
    if not years:
        raw = normalize_ocr_text(text)
        years = _VINTAGE_RE.findall(raw)
    flat: list[str] = []
    for y in years:
        if isinstance(y, tuple):
            flat.append("".join(y))
        else:
            flat.append(str(y))
    if not flat:
        fixed = normalize_ocr_text(text)
        flat = _VINTAGE_RE.findall(fixed)
    if not flat:
        return None
    return Counter(flat).most_common(1)[0][0]


def _sim(a: str, b: str) -> float:
    if not a or not b or fuzz is None:
        return 0.0
    a, b = normalize_compare_text(a), normalize_compare_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    return float(
        max(
            fuzz.token_set_ratio(a, b),
            fuzz.partial_ratio(a, b) * 0.95,
            fuzz.WRatio(a, b) * 0.9,
        )
    )


def _is_bad_producer_name(name: str | None) -> bool:
    if not name or not str(name).strip():
        return True
    n = normalize_ocr_text(str(name))
    if len(n) < 3:
        return True
    if _BAD_PRODUCER_RE.match(n):
        return True
    # «винодельня года» / award noise
    if "винодельн" in n and "год" in n:
        return True
    return False


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in normalize_ocr_text(text).split()
        if len(t) >= 3 and t not in _STOP_TOKENS and not t.isdigit()
    }


def _distinctive_tokens(text: str) -> set[str]:
    return {
        t
        for t in _tokens(text)
        if t not in _GENERIC_PRODUCER_TOKENS and len(t) >= 4
    }


def _token_hit(a: str, b: str) -> bool:
    """Fuzzy token overlap for OCR typos (литавщуков ≈ литавщук)."""
    a, b = normalize_compare_text(a), normalize_compare_text(b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    if fuzz is None:
        return False
    return float(fuzz.ratio(a, b)) >= 78.0


def _sets_overlap(a: set[str], b: set[str]) -> bool:
    for x in a:
        for y in b:
            if _token_hit(x, y):
                return True
    return False


def extract_color_key(text: str) -> str | None:
    raw = text or ""
    # Prefer explicit color words; last wins if several (rare)
    found: str | None = None
    for pat, key in _COLOR_PATTERNS:
        if pat.search(raw):
            found = key
    return found


def extract_type_key(text: str) -> str | None:
    raw = text or ""
    for pat, key in _TYPE_PATTERNS:
        if pat.search(raw):
            return key
    return None


_GRAPE_ALIASES = {
    "мерла": "мерло",
    "каберне-совиньон": "каберне совиньон",
}


def extract_grape_keys(text: str) -> set[str]:
    norm = normalize_ocr_text(text or "")
    if not norm:
        return set()
    hits: set[str] = set()
    toks = [t for t in norm.split() if len(t) >= 4]
    for g in _COMMON_GRAPES:
        gn = normalize_ocr_text(g)
        if len(gn) < 4:
            continue
        # multi-word: только явное вхождение
        if " " in gn:
            if gn in norm:
                hits.add(gn)
            continue
        if gn in toks or f" {gn} " in f" {norm} ":
            hits.add(gn)
            continue
        for tok in toks:
            if abs(len(tok) - len(gn)) > 2:
                continue
            if _token_hit(tok, gn):
                hits.add(gn)
                break
            if fuzz is not None and float(fuzz.ratio(tok, gn)) >= 78:
                hits.add(gn)
                break
    for raw, canon in _GRAPE_ALIASES.items():
        if raw in hits or raw in toks:
            hits.add(canon)
            hits.discard(raw)
    return hits


def catalog_color_keys(wine: Wine) -> set[str]:
    keys: set[str] = set()
    cat = normalize_ocr_text(wine.category or "")
    if cat in _CAT_TO_COLOR:
        keys.add(_CAT_TO_COLOR[cat])
    blob = f"{wine.name or ''} {wine.label or ''}"
    c = extract_color_key(blob)
    if c:
        keys.add(c)
    return keys


def catalog_type_key(wine: Wine) -> str | None:
    blob = f"{wine.name or ''} {wine.label or ''} {wine.category or ''}"
    return extract_type_key(blob)


def catalog_grape_keys(wine: Wine) -> set[str]:
    blob = f"{wine.grape_variety or ''} {wine.name or ''} {wine.label or ''}"
    keys = extract_grape_keys(blob)
    # also split grape_variety on commas
    gv = normalize_ocr_text(wine.grape_variety or "")
    for part in re.split(r"[,;/]| и ", gv):
        part = part.strip()
        if len(part) >= 4:
            keys.add(part)
            for g in _COMMON_GRAPES:
                gn = normalize_ocr_text(g)
                if gn and (gn in part or part in gn or _token_hit(part, gn)):
                    keys.add(gn)
    return keys


def _colors_compatible(ocr: str, catalog: set[str]) -> bool:
    if ocr in catalog:
        return True
    # «чёрный» мускатель ≈ красное
    if ocr == "black" and ("red" in catalog or "black" in catalog):
        return True
    if ocr == "red" and ("red" in catalog or "black" in catalog):
        return True
    return False


def _significant_label_lines(text: str) -> list[str]:
    """Непустые нормализованные строки этикетки (без совсем коротких / generic)."""
    out: list[str] = []
    for raw in (text or "").splitlines():
        ln = normalize_compare_text(normalize_ocr_text(raw))
        if len(ln) < 3:
            continue
        if ln in _GENERIC_LABEL_LINES:
            continue
        out.append(ln)
    return out


def _line_ends_conflict(a: str, b: str) -> bool:
    """Линии «не похожи»: отличаются и первые 2, и последние 2 буквы (без пробелов).

    Высокий fuzzy-sim / подстрока — не конфликт.
    """
    if not a or not b:
        return True
    if a == b or a in b or b in a:
        return False
    if _sim(a, b) >= _LINE_STRUCT_MATCH_MIN:
        return False
    aa = a.replace(" ", "")
    bb = b.replace(" ", "")
    if len(aa) < 4 or len(bb) < 4:
        return _sim(a, b) < 55.0
    return aa[:2] != bb[:2] and aa[-2:] != bb[-2:]


def label_lines_structure(
    ocr_text: str,
    catalog_label: str,
) -> dict[str, Any]:
    """Сравнение структуры строк OCR искомого и label каталога.

    weak=True если:
      - сильно разное число строк, и
      - совпала только часть строк, и
      - несовпавшие строки не похожи друг на друга (края линий).
    Тогда при cosine < _LINE_STRUCT_COS_MAX — hard reject другого вина.

    majority_unmatched_block=True если у более длинной стороны >3 значимых строк
    и >50% строк не совпали — разные вина (блокирующий фактор, без порога cos).
    """
    ocr = _significant_label_lines(ocr_text)
    cat = _significant_label_lines(catalog_label)
    n_ocr, n_cat = len(ocr), len(cat)
    empty = {
        "ocr_lines": n_ocr,
        "catalog_lines": n_cat,
        "matched": 0,
        "matched_ratio": 0.0,
        "line_count_weak": False,
        "partial": False,
        "unmatched_dissimilar": False,
        "weak": False,
        "majority_unmatched_block": False,
    }
    if n_ocr < 1 or n_cat < 1:
        return empty

    used_cat: set[int] = set()
    matched_ocr_idx: set[int] = set()
    for i, ol in enumerate(ocr):
        best_j = -1
        best_sc = -1.0
        for j, cl in enumerate(cat):
            if j in used_cat:
                continue
            sc = _sim(ol, cl)
            if sc > best_sc:
                best_sc = sc
                best_j = j
        if best_j < 0:
            continue
        cl = cat[best_j]
        if best_sc >= _LINE_STRUCT_MATCH_MIN or not _line_ends_conflict(ol, cl):
            used_cat.add(best_j)
            matched_ocr_idx.add(i)

    matched = len(matched_ocr_idx)
    unmatched_ocr = [ocr[i] for i in range(n_ocr) if i not in matched_ocr_idx]
    unmatched_cat = [cat[j] for j in range(n_cat) if j not in used_cat]

    min_n = min(n_ocr, n_cat)
    max_n = max(n_ocr, n_cat)
    line_count_weak = abs(n_ocr - n_cat) >= 2 or (
        min_n > 0 and (max_n / float(min_n)) >= 1.75
    )
    partial = matched < min_n
    matched_ratio = matched / float(max_n) if max_n else 0.0
    unmatched_ratio = 1.0 - matched_ratio
    # >3 строк на более длинной стороне и >50% строк не совпали → другое вино
    majority_unmatched_block = bool(max_n > 3 and unmatched_ratio > 0.5)

    pool = unmatched_cat if unmatched_cat else cat
    if not unmatched_ocr:
        unmatched_dissimilar = bool(unmatched_cat) and line_count_weak
    else:
        unmatched_dissimilar = all(
            all(_line_ends_conflict(ol, cl) for cl in pool) for ol in unmatched_ocr
        )

    weak = bool(
        n_ocr >= 2
        and n_cat >= 2
        and partial
        and unmatched_dissimilar
        and (
            line_count_weak
            or matched <= max(1, min_n // 2)
        )
        and matched_ratio < 0.6
    )
    return {
        "ocr_lines": n_ocr,
        "catalog_lines": n_cat,
        "matched": matched,
        "matched_ratio": round(matched_ratio, 4),
        "unmatched_ratio": round(unmatched_ratio, 4),
        "line_count_weak": line_count_weak,
        "partial": partial,
        "unmatched_dissimilar": unmatched_dissimilar,
        "weak": weak,
        "majority_unmatched_block": majority_unmatched_block,
        "unmatched_ocr": unmatched_ocr[:8],
        "unmatched_catalog": unmatched_cat[:8],
    }


def label_lines_reject_other_wine(
    ocr_text: str,
    catalog_label: str,
    cosine: float | None,
) -> tuple[bool, dict[str, Any]]:
    """True → другое вино: majority unmatched (>50%, >3 строк) или weak+cos<0.8."""
    info = label_lines_structure(ocr_text, catalog_label)
    cos = float(cosine) if cosine is not None else 0.0
    majority = bool(info.get("majority_unmatched_block"))
    weak_cos = bool(info.get("weak")) and cos < _LINE_STRUCT_COS_MAX
    reject = majority or weak_cos
    info = {
        **info,
        "cosine": round(cos, 6),
        "reject": reject,
        "reject_reason": (
            "majority_unmatched_lines"
            if majority
            else ("weak_structure_low_cosine" if weak_cos else None)
        ),
    }
    return reject, info


def brand_overlap_score(ocr_text: str, wine: Wine) -> float:
    """Доля отличительных OCR-токенов, найденных в name/winery/label каталога (0..1)."""
    ocr_toks = _tokens(ocr_text)
    cat_toks = _tokens(f"{wine.name or ''} {wine.winery or ''} {wine.label or ''}")
    if not ocr_toks or not cat_toks:
        return 0.0
    inter = ocr_toks & cat_toks
    # Dice coefficient — устойчивее к длинному OCR-мусору
    return (2.0 * len(inter)) / float(len(ocr_toks) + len(cat_toks))


def entities_from_llm_json(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Structured fields from Gemini/OpenAI JSON (preferred over raw-text heuristics)."""
    if not isinstance(data, dict) or not data:
        return None
    producer = data.get("producer")
    if _is_bad_producer_name(str(producer) if producer is not None else None):
        producer = None
    wine_name = str(data.get("wine_name") or "").strip()
    cupage = str(data.get("cupage") or "").strip()
    vintage = data.get("vintage_year") or data.get("vintage")
    if vintage is not None:
        vintage = str(vintage).strip() or None
        if vintage and not _VINTAGE_RE.fullmatch(vintage):
            vintage = extract_vintage(str(vintage))
    region = data.get("region")
    if region is not None:
        region = str(region).strip() or None
    wine = normalize_ocr_text(f"{wine_name} {cupage}".strip())
    full = str(data.get("full_text") or "")
    other = normalize_ocr_text(full) if full else wine
    if not wine and not producer and not vintage and not other:
        return None
    return {
        "producer": str(producer).strip() if producer else None,
        "producer_score": 95.0 if producer else 0.0,
        "wine": wine or None,
        "vintage": vintage,
        "region": region,
        "region_score": 80.0 if region else 0.0,
        "other": other,
        "text_norm": other or wine,
        "source": "llm_json",
        "wine_name": wine_name or None,
        "cupage": cupage or None,
    }


def merge_entities(
    text_ent: dict[str, Any],
    llm_ent: dict[str, Any] | None,
) -> dict[str, Any]:
    """LLM JSON overrides noisy text heuristics when fields are present."""
    out = dict(text_ent)
    if not llm_ent:
        return out
    if llm_ent.get("producer") and not _is_bad_producer_name(llm_ent.get("producer")):
        out["producer"] = llm_ent["producer"]
        out["producer_score"] = float(llm_ent.get("producer_score") or 95.0)
    if llm_ent.get("wine"):
        out["wine"] = llm_ent["wine"]
    if llm_ent.get("vintage"):
        out["vintage"] = llm_ent["vintage"]
    if llm_ent.get("region"):
        out["region"] = llm_ent["region"]
        out["region_score"] = float(llm_ent.get("region_score") or 80.0)
    llm_blob = " ".join(
        str(x)
        for x in (
            llm_ent.get("wine_name"),
            llm_ent.get("cupage"),
            llm_ent.get("text_norm"),
            (llm_ent.get("other") or ""),
        )
        if x
    )
    if llm_blob:
        if not out.get("color_key"):
            out["color_key"] = extract_color_key(llm_blob)
        if not out.get("type_key"):
            out["type_key"] = extract_type_key(llm_blob)
        g = extract_grape_keys(llm_blob)
        if g:
            out["grape_keys"] = sorted(set(out.get("grape_keys") or []) | g)
    if text_ent.get("text_norm"):
        out["text_norm"] = text_ent["text_norm"]
    elif llm_ent.get("text_norm"):
        out["text_norm"] = llm_ent["text_norm"]
    out["llm"] = {
        k: llm_ent.get(k)
        for k in ("producer", "wine", "vintage", "wine_name", "cupage")
    }
    out["source"] = "llm+text"
    return out


def extract_entities(
    text: str,
    *,
    producers: list[str],
    regions: list[str],
) -> dict[str, Any]:
    """Pull producer / vintage / region / color / type / grape from OCR text."""
    raw = text or ""
    norm = normalize_ocr_text(raw)
    vintage = extract_vintage(raw)
    ocr_distinct = _distinctive_tokens(raw)

    producer = None
    producer_score = 0.0
    for p in producers:
        if _is_bad_producer_name(p):
            continue
        pn = normalize_ocr_text(p)
        if len(pn) < 4:
            continue
        p_dist = _distinctive_tokens(p)
        # Без пересечения уникальных токенов — не берём (иначе «семейная винодельня»
        # матчит чужие хозяйства с тем же префиксом).
        if p_dist:
            if not ocr_distinct or not _sets_overlap(p_dist, ocr_distinct):
                continue
        else:
            # только общие слова в имени — слишком слабо
            continue
        sc = 0.0
        hit_n = sum(
            1
            for t in p_dist
            if any(_token_hit(t, o) for o in ocr_distinct)
        )
        for line in raw.splitlines():
            ln = normalize_ocr_text(line)
            if not ln:
                continue
            line_dist = _distinctive_tokens(line)
            if _sets_overlap(p_dist, line_dist):
                sc = max(sc, 88.0 + 4.0 * hit_n)
            p_core = " ".join(sorted(p_dist))
            l_core = " ".join(sorted(line_dist)) if line_dist else ln
            sc = max(sc, _sim(p_core, l_core), _sim(p_core, ln) * 0.85)
        sc = max(
            sc,
            _sim(" ".join(sorted(p_dist)), " ".join(sorted(ocr_distinct))),
            80.0 + 5.0 * hit_n,
        )
        # длинный уникальный токен в OCR важнее короткого совпадения
        if any(len(t) >= 6 and any(_token_hit(t, o) for o in ocr_distinct) for t in p_dist):
            sc = max(sc, 94.0)
        if sc > producer_score or (
            abs(sc - producer_score) < 3
            and hit_n > 0
            and len(pn) > len(normalize_ocr_text(producer or ""))
        ):
            producer_score = sc
            producer = p
    if producer_score < 60 or _is_bad_producer_name(producer):
        producer = None
        producer_score = 0.0

    region = None
    region_score = 0.0
    for r in regions:
        rn = normalize_ocr_text(r)
        if len(rn) < 3:
            continue
        sc = _sim(norm, rn)
        if sc > region_score:
            region_score = sc
            region = r
    if region_score < 60:
        region = None
        region_score = 0.0

    color_key = extract_color_key(raw)
    type_key = extract_type_key(raw)
    grape_keys = sorted(extract_grape_keys(raw))

    residual = norm
    for part in (producer, region, vintage):
        if not part:
            continue
        residual = residual.replace(normalize_ocr_text(part), " ")
    residual = re.sub(r"\s+", " ", residual).strip()

    return {
        "producer": producer,
        "producer_score": round(producer_score, 2),
        "wine": residual or None,
        "vintage": vintage,
        "region": region,
        "region_score": round(region_score, 2),
        "color_key": color_key,
        "type_key": type_key,
        "grape_keys": grape_keys,
        "other": norm,
        "text_norm": norm,
        "source": "text",
    }


def score_entities_against_wine(
    entities: dict[str, Any],
    wine: Wine,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Weighted TextScore in 0..1 against one catalog wine.

    Hard mismatch on key fields (winery/name/color/grape/type) — except vintage —
    forces score=0 (FinalScore must also be zeroed by caller).
    """
    w = weights or text_score_weights()
    name = wine.name or ""
    winery = wine.winery or ""
    region = wine.region or ""
    label = wine.label or ""
    catalog_blob = f"{name} {winery} {label}"
    catalog_vintage = extract_vintage(catalog_blob)

    wine_query = " ".join(
        x
        for x in (
            entities.get("wine") or "",
            str((entities.get("llm") or {}).get("wine_name") or ""),
            str((entities.get("llm") or {}).get("cupage") or ""),
        )
        if x
    ).strip() or (entities.get("wine") or "")

    wine_sim = max(
        _sim(wine_query, name),
        _sim(wine_query, label),
        _sim(entities.get("other") or "", name),
        _sim(entities.get("text_norm") or "", name),
    )
    producer_q = entities.get("producer") or ""
    producer_sim = max(
        _sim(producer_q, winery),
        _sim(producer_q, label),
        _sim(entities.get("other") or "", winery),
    )
    if producer_q and not _is_bad_producer_name(str(producer_q)):
        producer_sim = max(producer_sim, _sim(producer_q, catalog_blob))
        pq_dist = _distinctive_tokens(str(producer_q))
        winery_dist = _distinctive_tokens(winery) | _distinctive_tokens(label)
        if pq_dist and winery_dist and _sets_overlap(pq_dist, winery_dist):
            producer_sim = max(producer_sim, 92.0)
        elif pq_dist and winery_dist and not _sets_overlap(pq_dist, winery_dist):
            producer_sim = min(producer_sim, 35.0)

    # Vintage: soft only — другой год ≠ другое вино
    ev = entities.get("vintage")
    if ev and catalog_vintage:
        vintage_sim = 100.0 if str(ev) == str(catalog_vintage) else 42.0
    elif ev and not catalog_vintage:
        vintage_sim = 55.0
    elif catalog_vintage and not ev:
        vintage_sim = 50.0
    else:
        vintage_sim = 50.0

    region_sim = max(
        _sim(entities.get("region") or "", region),
        _sim(entities.get("other") or "", region),
    )
    other_sim = _sim(
        entities.get("other") or "",
        normalize_ocr_text(catalog_blob),
    )
    brand_sim = brand_overlap_score(
        entities.get("text_norm") or entities.get("other") or wine_query,
        wine,
    )

    ocr_color = entities.get("color_key") or extract_color_key(
        entities.get("text_norm") or entities.get("other") or ""
    )
    cat_colors = catalog_color_keys(wine)
    if ocr_color and cat_colors:
        color_sim = 100.0 if _colors_compatible(str(ocr_color), cat_colors) else 0.0
    elif ocr_color or cat_colors:
        color_sim = 50.0
    else:
        color_sim = 50.0

    ocr_type = entities.get("type_key") or extract_type_key(
        entities.get("text_norm") or entities.get("other") or ""
    )
    cat_type = catalog_type_key(wine)
    if ocr_type and cat_type:
        type_sim = 100.0 if ocr_type == cat_type else 0.0
    elif ocr_type or cat_type:
        type_sim = 50.0
    else:
        type_sim = 50.0

    ocr_grapes = set(entities.get("grape_keys") or [])
    if not ocr_grapes:
        ocr_grapes = extract_grape_keys(
            entities.get("text_norm") or entities.get("other") or wine_query
        )
    cat_grapes = catalog_grape_keys(wine)
    if ocr_grapes and cat_grapes:
        grape_sim = 100.0 if _sets_overlap(ocr_grapes, cat_grapes) else 0.0
    elif ocr_grapes or cat_grapes:
        grape_sim = 50.0
    else:
        grape_sim = 50.0

    parts = {
        "wine": wine_sim / 100.0,
        "producer": producer_sim / 100.0,
        "vintage": vintage_sim / 100.0,
        "region": region_sim / 100.0,
        "other": other_sim / 100.0,
        "brand": float(brand_sim),
        "color": color_sim / 100.0,
        "grape": grape_sim / 100.0,
        "type": type_sim / 100.0,
    }
    total_w = sum(w.get(k, 0.0) for k in parts) or 1.0
    score = sum(parts[k] * w.get(k, 0.0) for k in parts) / total_w

    mismatches: list[str] = []
    if (
        producer_q
        and not _is_bad_producer_name(str(producer_q))
        and float(entities.get("producer_score") or 0) >= 70
        and producer_sim < _KEY_CONFLICT_MAX
    ):
        mismatches.append("winery")
    name_toks = _distinctive_tokens(name)
    ocr_name_toks = _distinctive_tokens(wine_query) - _distinctive_tokens(
        str(producer_q or "")
    )
    if (
        name_toks
        and ocr_name_toks
        and wine_sim < _KEY_CONFLICT_MAX
        and not _sets_overlap(name_toks, ocr_name_toks)
        and any(len(t) >= 5 for t in ocr_name_toks)
    ):
        mismatches.append("name")
    if ocr_color and cat_colors and color_sim < _KEY_CONFLICT_MAX:
        mismatches.append("color")
    if ocr_type and cat_type and type_sim < _KEY_CONFLICT_MAX:
        mismatches.append("type")
    if ocr_grapes and cat_grapes and grape_sim < _KEY_CONFLICT_MAX:
        mismatches.append("grape")

    hard_mismatch = bool(mismatches)
    if hard_mismatch:
        score = 0.0
    else:
        present_ok = 0
        present_n = 0
        for flag, sim in (
            (
                bool(producer_q) and float(entities.get("producer_score") or 0) >= 70,
                producer_sim,
            ),
            (bool(str(wine_query).strip()), wine_sim),
            (bool(ocr_color) and bool(cat_colors), color_sim),
            (bool(ocr_type) and bool(cat_type), type_sim),
            (bool(ocr_grapes) and bool(cat_grapes), grape_sim),
        ):
            if not flag:
                continue
            present_n += 1
            if sim >= _KEY_MATCH_MIN:
                present_ok += 1
        if present_n >= 2 and present_ok == present_n:
            score = max(score, 0.92)
            if present_n >= 3:
                score = max(score, 0.97)
        elif (
            present_n >= 2
            and producer_sim >= _KEY_MATCH_MIN
            and wine_sim >= _KEY_MATCH_MIN
            and present_ok >= 2
        ):
            score = max(score, 0.78)

    return {
        "score": round(score, 4),
        "parts": {k: round(v, 4) for k, v in parts.items()},
        "weights": dict(w),
        "catalog_vintage": catalog_vintage,
        "hard_mismatch": hard_mismatch,
        "mismatch_fields": mismatches,
        "keys": {
            "ocr_color": ocr_color,
            "catalog_colors": sorted(cat_colors),
            "ocr_type": ocr_type,
            "catalog_type": cat_type,
            "ocr_grapes": sorted(ocr_grapes),
            "catalog_grapes": sorted(cat_grapes)[:12],
        },
    }


def compute_final_score(
    *,
    text_score_01: float,
    cosine: float | None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """FinalScore = w_text×TextScore + w_cos×Cosine (оба 0..1)."""
    fw = weights or final_score_weights()
    wt = float(fw.get("text") or 0.0)
    wc = float(fw.get("cosine") or 0.0)
    s = max(0.0, wt) + max(0.0, wc)
    if s <= 0:
        wt, wc, s = 0.45, 0.55, 1.0
    wt, wc = wt / s, wc / s
    cos = float(cosine) if cosine is not None else 0.0
    cos = max(0.0, min(1.0, cos))
    ts = max(0.0, min(1.0, float(text_score_01)))
    final = wt * ts + wc * cos
    return {
        "final_score": round(final, 6),
        "text_score_01": round(ts, 4),
        "cosine": round(cos, 6),
        "weights": {"text": round(wt, 4), "cosine": round(wc, 4)},
    }


def _ensemble_entities(variant_entities: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Vote across OCR variants: majority vintage; best producer/region by score."""
    vintages = [
        e.get("vintage")
        for e in variant_entities.values()
        if e.get("vintage")
    ]
    vintage = Counter(vintages).most_common(1)[0][0] if vintages else None

    best_prod = None
    best_ps = -1.0
    best_reg = None
    best_rs = -1.0
    wine_bits: list[str] = []
    other_bits: list[str] = []
    for e in variant_entities.values():
        prod = e.get("producer")
        if (
            (e.get("producer_score") or 0) > best_ps
            and prod
            and not _is_bad_producer_name(str(prod))
        ):
            best_ps = float(e["producer_score"])
            best_prod = prod
        if (e.get("region_score") or 0) > best_rs and e.get("region"):
            best_rs = float(e["region_score"])
            best_reg = e["region"]
        if e.get("wine"):
            wine_bits.append(str(e["wine"]))
        if e.get("other"):
            other_bits.append(str(e["other"]))

    wine_joined = normalize_ocr_text(" ".join(wine_bits))
    other_joined = normalize_ocr_text(" ".join(other_bits))
    color_votes = [
        e.get("color_key") for e in variant_entities.values() if e.get("color_key")
    ]
    type_votes = [
        e.get("type_key") for e in variant_entities.values() if e.get("type_key")
    ]
    grape_all: set[str] = set()
    for e in variant_entities.values():
        grape_all.update(e.get("grape_keys") or [])
        grape_all |= extract_grape_keys(e.get("text_norm") or e.get("other") or "")
    color_key = Counter(color_votes).most_common(1)[0][0] if color_votes else None
    type_key = Counter(type_votes).most_common(1)[0][0] if type_votes else None
    if not color_key:
        color_key = extract_color_key(other_joined)
    if not type_key:
        type_key = extract_type_key(other_joined)
    return {
        "producer": best_prod,
        "producer_score": round(best_ps, 2) if best_prod else 0.0,
        "wine": wine_joined or None,
        "vintage": vintage,
        "region": best_reg,
        "region_score": round(best_rs, 2) if best_reg else 0.0,
        "color_key": color_key,
        "type_key": type_key,
        "grape_keys": sorted(grape_all),
        "other": other_joined,
        "text_norm": other_joined,
        "vintage_votes": len(vintages),
        "source": "ensemble",
    }


def get_producers_regions_cached(db: Session) -> tuple[list[str], list[str], dict[str, Any]]:
    """Уникальные winery/region из каталога; кеш по числу вин."""
    from sqlalchemy import func

    global _PRODUCERS_REGIONS_CACHE
    n_docs = int(db.scalar(select(func.count()).select_from(Wine)) or 0)
    if (
        _PRODUCERS_REGIONS_CACHE.get("n_docs") == n_docs
        and _PRODUCERS_REGIONS_CACHE.get("producers") is not None
    ):
        return (
            list(_PRODUCERS_REGIONS_CACHE["producers"]),
            list(_PRODUCERS_REGIONS_CACHE["regions"]),
            {"cached": True, "n_docs": n_docs},
        )
    producers = sorted(
        {
            s.strip()
            for s in db.scalars(select(Wine.winery).where(Wine.winery.is_not(None))).all()
            if isinstance(s, str) and s.strip() and not _is_bad_producer_name(s)
        }
    )
    regions = sorted(
        {
            s.strip()
            for s in db.scalars(select(Wine.region).where(Wine.region.is_not(None))).all()
            if isinstance(s, str) and s.strip()
        }
    )
    _PRODUCERS_REGIONS_CACHE = {
        "n_docs": n_docs,
        "producers": producers,
        "regions": regions,
    }
    return producers, regions, {"cached": False, "n_docs": n_docs}


def warm_producers_regions_cache(db: Session) -> dict[str, Any]:
    """Прогрев кеша при старте backend."""
    _p, _r, meta = get_producers_regions_cached(db)
    return {
        **meta,
        "n_producers": len(_p),
        "n_regions": len(_r),
    }


def evaluate_ocr_wine_id_help(
    db: Session,
    *,
    ocr_step: dict[str, Any],
    visual_candidate_ids: list[int],
    cosine_by_id: dict[int, float] | None = None,
    top_k: int = 20,
    exclusive_reject_ids: set[int] | frozenset[int] | None = None,
    compute_fin1: bool = True,
    compute_fin2: bool = True,
    fast_text_match: bool = False,
) -> dict[str, Any]:
    """TextScore + FinalScore vs unique embedding candidates (SigLIP2∪DINOv3).

    fast_text_match: Soft IDF/R без CMS name; кеш producers/regions;
    IDF — веса редкости по каталогу, скоринг только по visual top-N.
    """
    t0 = time.perf_counter()
    weights = text_score_weights()
    fweights = final_score_weights()
    cos_map = {int(k): float(v) for k, v in (cosine_by_id or {}).items()}
    exclusive_ids = {int(x) for x in (exclusive_reject_ids or set())}
    explain = {
        "formula": (
            "TextScore = Σ w_i×Sim_i / Σw  "
            "(wine, producer, color, type/купаж=grape, vintage, region, other, brand); "
            "hard mismatch на ключе ≠ год → TextScore=FinalScore=0"
        ),
        "final_formula": (
            "FinalScore = w_ocr×TextScore + w_emb×Cosine  "
            f"(w_ocr/text={fweights['text']}, w_emb/cosine={fweights['cosine']}); "
            "hard_mismatch → FinalScore=0"
        ),
        "weights": weights,
        "final_weights": fweights,
        "notes": [
            "Ключевые поля TextScore: винодельня, название, цвет, тип, купаж/сорта, год.",
            "Веса полей и вклад OCR/embedding задаются в Настройки → вкладка «Веса».",
            "Hard mismatch: ключ есть на этикетке и в карточке, но не совпал "
            "(кроме года) → score 0, cosine не спасает.",
            "Блок label_lines (без порога cos): если у более длинной этикетки >3 значимых "
            "строк и >50% строк не совпали (matched/max < 0.5) → hard mismatch "
            "(разные вина).",
            "Слабая структура строк OCR↔label при cosine<0.8 → hard mismatch label_lines.",
            "Год: только soft-штраф при расхождении (то же вино другого урожая).",
            "FinalScore ранжирует visual-кандидатов канала. "
            "matched_wine берётся из выбранного в настройках Final score "
            "(fin1 / fin2 / XGBoost / Cross Encoder).",
            "TextScore2 / FinalScore2 (Soft TF-IDF): только Google Vision и Yandex OCR.",
            "Soft veto: reject FinalScore(v1) winner if Soft cover_Q / TextScore2 слишком низкие.",
        ],
    }

    if fuzz is None:
        return {"ok": False, "error": "rapidfuzz не установлен", "ms": 0, "explain": explain}

    visual_set: list[int] = []
    seen: set[int] = set()
    for raw in visual_candidate_ids or []:
        wid = int(raw)
        if wid in seen:
            continue
        seen.add(wid)
        visual_set.append(wid)
    visual_lookup = set(visual_set)

    candidate_wines: list[Wine] = []
    if visual_set:
        rows = list(db.scalars(select(Wine).where(Wine.id.in_(visual_set))).all())
        by_fetched = {w.id: w for w in rows}
        candidate_wines = [by_fetched[i] for i in visual_set if i in by_fetched]

    # Soft TF-IDF IDF — веса редкости по каталогу (cached); скоринг ≠ весь каталог
    soft_idf, soft_default_idf, soft_idf_meta = build_catalog_idf(
        db, exclude_name=bool(fast_text_match)
    )
    soft_by_channel: dict[str, list[dict[str, Any]]] = {}

    producers, regions, prod_meta = get_producers_regions_cached(db)

    variants = (ocr_step or {}).get("variants") or {}
    per_variant: dict[str, Any] = {}
    variant_entities: dict[str, dict[str, Any]] = {}

    prep_keys = ("A", "B", "C", "D")
    eng_keys = ("rapid", "tess", "easy", "surya")
    preferred = [f"{p}_{e}" for p in prep_keys for e in eng_keys]
    keys = [k for k in preferred if k in variants]
    if not keys:
        keys = [k for k in prep_keys if k in variants]
    if "gemini" in variants and "gemini" not in keys:
        keys.append("gemini")
    if "openai" in variants and "openai" not in keys:
        keys.append("openai")
    for name in ("deepseek", "qwen", "yandex", "google_vision"):
        if name in variants and name not in keys:
            keys.append(name)

    def _score_against_candidates(
        entities: dict[str, Any],
        *,
        ocr_raw_text: str | None = None,
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        ocr_blob = str(
            ocr_raw_text
            or entities.get("ocr_raw")
            or entities.get("text_norm")
            or entities.get("other")
            or entities.get("wine")
            or ""
        )
        for wrow in candidate_wines:
            detail = score_entities_against_wine(entities, wrow, weights=weights)
            cos = cos_map.get(int(wrow.id))
            hard = bool(detail.get("hard_mismatch"))
            mismatches = list(detail.get("mismatch_fields") or [])
            line_info: dict[str, Any] | None = None
            if int(wrow.id) in exclusive_ids:
                hard = True
                if "exclusive_lexicon" not in mismatches:
                    mismatches.append("exclusive_lexicon")
            if not hard and (wrow.label or "").strip() and ocr_blob.strip():
                reject_lines, line_info = label_lines_reject_other_wine(
                    ocr_blob,
                    str(wrow.label or ""),
                    cos,
                )
                if reject_lines:
                    hard = True
                    if "label_lines" not in mismatches:
                        mismatches.append("label_lines")
            if hard:
                final = {
                    "final_score": 0.0,
                    "text_score_01": 0.0,
                    "cosine": None if cos is None else round(float(cos), 6),
                    "weights": fweights,
                    "hard_mismatch": True,
                    "mismatch_fields": mismatches,
                }
            else:
                final = compute_final_score(
                    text_score_01=detail["score"],
                    cosine=cos,
                    weights=fweights,
                )
                if not compute_fin1:
                    final = {**final, "final_score": 0.0, "skipped": True}
            scored.append(
                {
                    "id": wrow.id,
                    "score": round(0.0 if hard else detail["score"] * 100, 2),
                    "score_01": 0.0 if hard else detail["score"],
                    "parts": detail["parts"],
                    "name": wrow.name,
                    "winery": wrow.winery,
                    "cosine": None if cos is None else round(float(cos), 6),
                    "final_score": final["final_score"],
                    "final": final,
                    "hard_mismatch": hard,
                    "mismatch_fields": mismatches,
                    "keys": detail.get("keys"),
                    "label_lines": line_info,
                }
            )
        scored.sort(
            key=lambda x: (
                -float(x.get("final_score") or 0),
                -float(x.get("score") or 0),
                x["id"],
            )
        )
        return scored

    for key in keys:
        v = variants.get(key) or {}
        if v.get("alias_of"):
            continue
        raw_text = v.get("text") or ""
        entry: dict[str, Any] = {
            "id": key,
            "name": v.get("name"),
            "engine": v.get("engine"),
            "preprocess": v.get("preprocess"),
            "ok": bool(v.get("ok") and raw_text.strip()),
        }
        if not entry["ok"]:
            entry.update(
                {
                    "entities": None,
                    "top_matches": [],
                    "top_wine_ids": [],
                    "visual_support": [],
                    "best_visual": None,
                    "best_final": None,
                }
            )
            per_variant[key] = entry
            continue

        text_ent = extract_entities(raw_text, producers=producers, regions=regions)
        llm_ent = entities_from_llm_json(
            v.get("json") if isinstance(v.get("json"), dict) else None
        )
        entities = merge_entities(text_ent, llm_ent)
        entities["ocr_raw"] = raw_text
        variant_entities[key] = entities
        entry["entities"] = entities
        entry["query_norm"] = entities.get("text_norm")

        support = _score_against_candidates(entities, ocr_raw_text=raw_text)

        # Algorithm 2 (Soft TF-IDF): Google Vision + Yandex only
        if key in SOFT_CHANNELS and candidate_wines and compute_fin2:
            soft_rows = score_channel_soft(
                query_text=raw_text,
                wines=candidate_wines,
                cosine_by_id=cos_map,
                idf=soft_idf,
                default_idf=soft_default_idf,
                fweights=fweights,
                exclusive_reject_ids=exclusive_ids,
                exclude_name=bool(fast_text_match),
            )
            soft_by_channel[key] = soft_rows
            by_soft = {int(r["id"]): r for r in soft_rows}
            for row in support:
                s2 = by_soft.get(int(row["id"]))
                if not s2:
                    continue
                # Exclusive / other hard reject must also zero Soft FinalScore2
                if row.get("hard_mismatch") or s2.get("exclusive_rejected"):
                    row["score2"] = 0.0
                    row["score2_01"] = 0.0
                    row["final_score2"] = 0.0
                    row["soft_tfidf"] = s2.get("soft_tfidf")
                    row["exclusive_rejected"] = True
                    if row.get("hard_mismatch"):
                        pass
                    else:
                        row["hard_mismatch"] = True
                        mm = list(row.get("mismatch_fields") or [])
                        if "exclusive_lexicon" not in mm:
                            mm.append("exclusive_lexicon")
                        row["mismatch_fields"] = mm
                        row["score"] = 0.0
                        row["score_01"] = 0.0
                        row["final_score"] = 0.0
                else:
                    row["score2"] = s2["score2"]
                    row["score2_01"] = s2["score2_01"]
                    row["final_score2"] = s2["final_score2"]
                    row["soft_tfidf"] = s2.get("soft_tfidf")
            soft_sorted = sorted(
                support,
                key=lambda x: (
                    -float(x.get("final_score2") or 0),
                    -float(x.get("score2") or 0),
                    x["id"],
                ),
            )
            entry["best_final2"] = next(
                (
                    r
                    for r in soft_sorted
                    if not r.get("hard_mismatch")
                    and float(r.get("final_score2") or 0) > 0
                ),
                None,
            )
            entry["visual_support2"] = soft_sorted[:top_k]

        top = support[:top_k]
        entry["visual_support"] = top
        entry["top_matches"] = top
        entry["top_wine_ids"] = [s["id"] for s in top]
        # best by TextScore alone (для бейджей G/O)
        by_text = sorted(support, key=lambda x: (-x["score"], x["id"]))
        entry["best_visual"] = by_text[0] if by_text else None
        entry["best_final"] = top[0] if top else None
        if visual_set:
            top1 = visual_set[0]
            try:
                entry["visual_top1_ocr_rank"] = entry["top_wine_ids"].index(top1) + 1
            except ValueError:
                entry["visual_top1_ocr_rank"] = None
            entry["visual_top1_support_score"] = next(
                (s["score"] for s in support if s["id"] == top1), None
            )
        per_variant[key] = entry

    ensemble_ent = _ensemble_entities(variant_entities) if variant_entities else {}
    ensemble_block: dict[str, Any] = {"entities": ensemble_ent, "ok": bool(ensemble_ent)}
    if ensemble_ent:
        ens_raw = "\n".join(
            str((variant_entities[k] or {}).get("ocr_raw") or "")
            for k in variant_entities
            if (variant_entities[k] or {}).get("ocr_raw")
        )
        support_e = _score_against_candidates(
            ensemble_ent, ocr_raw_text=ens_raw or None
        )
        top_e = support_e[:top_k]
        ensemble_block["visual_support"] = top_e
        ensemble_block["top_matches"] = top_e
        ensemble_block["top_wine_ids"] = [s["id"] for s in top_e]
        by_text_e = sorted(support_e, key=lambda x: (-x["score"], x["id"]))
        ensemble_block["best_visual"] = by_text_e[0] if by_text_e else None
        ensemble_block["best_final"] = top_e[0] if top_e else None

    # Лучший OCR-канал по TextScore (как раньше для UI)
    best_key = None
    best_score = -1.0
    for key, entry in per_variant.items():
        bv = entry.get("best_visual") or {}
        sc = float(bv.get("score") or 0)
        if sc > best_score:
            best_score = sc
            best_key = key
    if ensemble_block.get("best_visual"):
        esc = float(ensemble_block["best_visual"]["score"])
        if esc >= best_score:
            best_key = "ensemble"
            best_score = esc

    # Итоговый ранг по FinalScore: берём лучший text среди каналов на каждый wine_id,
    # затем смешиваем с cosine.
    best_text_by_id: dict[int, dict[str, Any]] = {}
    for key, entry in list(per_variant.items()) + (
        [("ensemble", ensemble_block)] if ensemble_block.get("ok") else []
    ):
        for row in entry.get("visual_support") or []:
            wid = int(row["id"])
            prev = best_text_by_id.get(wid)
            if prev is None or float(row["score_01"]) > float(prev["score_01"]):
                best_text_by_id[wid] = {
                    "id": wid,
                    "score": row["score"],
                    "score_01": row["score_01"],
                    "parts": row.get("parts"),
                    "name": row.get("name"),
                    "winery": row.get("winery"),
                    "text_from": key,
                    "hard_mismatch": row.get("hard_mismatch"),
                    "mismatch_fields": row.get("mismatch_fields"),
                    "keys": row.get("keys"),
                    "label_lines": row.get("label_lines"),
                    "score2": row.get("score2"),
                    "score2_01": row.get("score2_01"),
                    "final_score2": row.get("final_score2"),
                    "soft_tfidf": row.get("soft_tfidf"),
                }

    final_ranked: list[dict[str, Any]] = []
    for wid, row in best_text_by_id.items():
        cos = cos_map.get(wid)
        hard = bool(row.get("hard_mismatch"))
        if hard:
            final = {
                "final_score": 0.0,
                "text_score_01": 0.0,
                "cosine": None if cos is None else round(float(cos), 6),
                "weights": fweights,
                "hard_mismatch": True,
                "mismatch_fields": row.get("mismatch_fields") or [],
            }
        else:
            final = compute_final_score(
                text_score_01=float(row["score_01"]),
                cosine=cos,
                weights=fweights,
            )
            if not compute_fin1:
                final = {**final, "final_score": 0.0, "skipped": True}
        final_ranked.append(
            {
                **row,
                "cosine": None if cos is None else round(float(cos), 6),
                "final_score": final["final_score"],
                "final": final,
            }
        )
    final_ranked.sort(
        key=lambda x: (
            0 if not x.get("hard_mismatch") and float(x.get("final_score") or 0) > 0 else 1,
            -float(x["final_score"]),
            -float(x["score_01"]),
            x["id"],
        )
    )

    id_counts: dict[int, int] = {}
    for entry in per_variant.values():
        for wid in set(entry.get("top_wine_ids") or []):
            id_counts[wid] = id_counts.get(wid, 0) + 1
    agreement = sorted(
        (
            {"id": wid, "variant_hits": cnt, "in_visual": wid in visual_lookup}
            for wid, cnt in id_counts.items()
            if cnt >= 2
        ),
        key=lambda x: (-x["variant_hits"], x["id"]),
    )[:20]

    best_final = next(
        (
            r
            for r in final_ranked
            if not r.get("hard_mismatch") and float(r.get("final_score") or 0) > 0
        ),
        None,
    )
    ocr_rejected_all = best_final is None and bool(final_ranked)

    # --- Algorithm 2 aggregate: best TextScore2 across Soft channels (Yandex / GV) ---
    best_text2_by_id: dict[int, dict[str, Any]] = {}
    for ch, rows in soft_by_channel.items():
        for row in rows:
            wid = int(row["id"])
            if wid in exclusive_ids or row.get("exclusive_rejected") or row.get(
                "hard_mismatch"
            ):
                # Keep a zeroed stub so veto / UI can see exclusive reject
                prev = best_text2_by_id.get(wid)
                if prev is not None and float(prev.get("final_score2") or 0) > 0:
                    continue
                best_text2_by_id[wid] = {
                    "id": wid,
                    "score2": 0.0,
                    "score2_01": 0.0,
                    "final_score2": 0.0,
                    "cosine": row.get("cosine"),
                    "name": row.get("name"),
                    "winery": row.get("winery"),
                    "text_from": ch,
                    "soft_tfidf": row.get("soft_tfidf"),
                    "exclusive_rejected": True,
                    "hard_mismatch": True,
                    "mismatch_fields": ["exclusive_lexicon"],
                }
                continue
            prev = best_text2_by_id.get(wid)
            if prev is None or float(row["score2_01"]) > float(prev.get("score2_01") or 0):
                best_text2_by_id[wid] = {
                    "id": wid,
                    "score2": row["score2"],
                    "score2_01": row["score2_01"],
                    "final_score2": row["final_score2"],
                    "cosine": row.get("cosine"),
                    "name": row.get("name"),
                    "winery": row.get("winery"),
                    "text_from": ch,
                    "soft_tfidf": row.get("soft_tfidf"),
                }

    # Attach Soft scores onto FinalScore(v1) ranked rows when missing
    for r in final_ranked:
        wid = int(r["id"])
        soft_row = best_text2_by_id.get(wid)
        if soft_row is None:
            continue
        if r.get("hard_mismatch") or soft_row.get("exclusive_rejected"):
            r["score2"] = 0.0
            r["score2_01"] = 0.0
            r["final_score2"] = 0.0
            r["soft_tfidf"] = soft_row.get("soft_tfidf")
            r["exclusive_rejected"] = True
            continue
        if r.get("score2_01") is None:
            r["score2"] = soft_row.get("score2")
            r["score2_01"] = soft_row.get("score2_01")
            r["final_score2"] = soft_row.get("final_score2")
            r["soft_tfidf"] = soft_row.get("soft_tfidf")

    final_ranked2: list[dict[str, Any]] = [
        r
        for r in best_text2_by_id.values()
        if not r.get("exclusive_rejected")
        and not r.get("hard_mismatch")
        and int(r["id"]) not in exclusive_ids
        and float(r.get("final_score2") or 0) > 0
    ]
    final_ranked2.sort(
        key=lambda x: (
            -float(x.get("final_score2") or 0),
            -float(x.get("score2") or 0),
            -float((x.get("soft_tfidf") or {}).get("cover_q") or 0),
            x["id"],
        )
    )
    best_final2 = final_ranked2[0] if final_ranked2 else None
    # Require minimal informative text + cover
    if best_final2 is not None:
        st = best_final2.get("soft_tfidf") or {}
        if float(best_final2.get("final_score2") or 0) < 0.50 or float(
            st.get("cover_q") or 0
        ) < 0.45:
            best_final2 = None

    # Soft TF-IDF veto on FinalScore(v1) winner
    soft_veto: dict[str, Any] | None = None
    if best_final is not None and best_text2_by_id:
        wid = int(best_final["id"])
        soft_row = best_text2_by_id.get(wid)
        if soft_row is not None:
            stf = soft_row.get("soft_tfidf") or {}
            cq = float(stf.get("cover_q") or 0)
            s2 = float(soft_row.get("score2_01") or 0)
            best_final["score2"] = soft_row.get("score2")
            best_final["score2_01"] = soft_row.get("score2_01")
            best_final["final_score2"] = soft_row.get("final_score2")
            best_final["soft_tfidf"] = stf
            if cq < SOFT_VETO_COVER_Q or s2 < SOFT_VETO_SCORE2:
                soft_veto = {
                    "wine_id": wid,
                    "cover_q": round(cq, 6),
                    "score2_01": round(s2, 6),
                    "thresholds": {
                        "cover_q": SOFT_VETO_COVER_Q,
                        "score2_01": SOFT_VETO_SCORE2,
                    },
                    "reason": (
                        "Soft TF-IDF: OCR почти не покрывает текст этикетки кандидата "
                        f"(cover_Q={cq:.3f} < {SOFT_VETO_COVER_Q} или "
                        f"TextScore2={s2:.3f} < {SOFT_VETO_SCORE2})"
                    ),
                }
                best_final = None
                ocr_rejected_all = True

    explain["soft_tfidf"] = {
        "formula": (
            "TextScore2 = 0.80·cover_Q + 0.20·max(partial,0.5·ngram3) "
            "+ 0.08·uniq_recall; "
            "cover_Q = IDF-weighted soft token coverage of query by catalog "
            "(Jaro–Winkler); cover_R diagnostic only; "
            "channels = google_vision, yandex"
        ),
        "final_formula": (
            "FinalScore2 = w_ocr×TextScore2 + w_emb×Cosine "
            f"(same weights as FinalScore: text={fweights['text']}, "
            f"cosine={fweights['cosine']})"
        ),
        "veto": (
            f"Reject FinalScore(v1) winner if Soft cover_Q < {SOFT_VETO_COVER_Q} "
            f"or TextScore2 < {SOFT_VETO_SCORE2} (Yandex/Google Vision only)"
        ),
        "exclusive": (
            "exclusive_lexicon reject → TextScore2=FinalScore2=0; "
            "best_final2 skips rejected ids (same as FinalScore v1)"
        ),
        "idf": soft_idf_meta,
        "channels": sorted(SOFT_CHANNELS),
        "exclude_name": bool(fast_text_match),
        "note": (
            "IDF — веса редкости токенов по каталогу (не сравнение со всеми винами); "
            "Soft score только vs visual top-N. "
            + (
                "fast: без CMS name/region в R и корпусе IDF."
                if fast_text_match
                else "legacy: R включает name/region."
            )
        ),
        "producers_regions": prod_meta,
    }

    return {
        "ok": True,
        "explain": explain,
        "weights": weights,
        "final_weights": fweights,
        "per_variant": per_variant,
        "ensemble": ensemble_block,
        "best_variant_for_visual_support": best_key,
        "best_visual_support_score": round(best_score, 2) if best_key else None,
        "final_ranked": final_ranked[:top_k],
        "best_final": best_final,
        "best_wine_id": int(best_final["id"]) if best_final else None,
        "final_ranked2": final_ranked2[:top_k],
        "best_final2": best_final2,
        "best_wine_id2": int(best_final2["id"]) if best_final2 else None,
        "ocr_rejected_all": ocr_rejected_all,
        "soft_veto": soft_veto,
        "variant_agreement_top": agreement,
        "visual_candidate_ids": visual_set,
        "candidate_count": len(candidate_wines),
        "fast_text_match": bool(fast_text_match),
        "path": "fast" if fast_text_match else "legacy",
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }
