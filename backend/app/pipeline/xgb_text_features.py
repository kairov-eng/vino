"""Нормализация текста и попарные признаки (OCR запроса vs эталон вина).

Общий модуль для prepare_dataset.py (offline) и будущего production-inference
XGBoost-реранкера. Только Python + rapidfuzz, без БД.

Пара = (ocr_text, WineRef, siglip_cosine). Признаки считаются на уровне группы
кандидатов одного скана (`group_features`), потому что часть признаков
относительная: ранг по cosine, отрыв от лучшего кандидата и т.п.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from app.pipeline.text_compare import (
    fix_digit_letter_homoglyphs,
    fix_latin_cyrillic_homoglyphs,
    normalize_compare_text,
)

FEATURE_VERSION = "1.4.0"

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_YEAR_RE = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")
_NUM_RE = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)(?![\d.,])")
_ALC_RE = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*%")
_VOL_RE = re.compile(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(мл|ml|л\b|l\b)", re.IGNORECASE)

_COLOR_WORDS: dict[str, set[str]] = {
    "red": {"красное", "красный", "красн", "red", "rouge", "rosso", "tinto", "rot"},
    "white": {"белое", "белый", "бел", "white", "blanc", "bianco", "blanco", "weiss"},
    "rose": {"розовое", "розовый", "розе", "rose", "rosé", "rosato", "rosado"},
    "orange": {"оранжевое", "оранж", "orange"},
}
_COLOR_CATEGORY_PREFIX = (
    ("красн", "red"),
    ("бел", "white"),
    ("розов", "rose"),
    ("оранж", "orange"),
)
_SWEET_WORDS: dict[str, set[str]] = {
    "dry": {"сухое", "сухой", "dry", "sec", "secco", "seco", "trocken", "brut", "брют"},
    "semi_dry": {"полусухое", "полусухой", "semi-dry", "demi-sec", "abboccato", "halbtrocken"},
    "semi_sweet": {"полусладкое", "полусладкий", "semi-sweet", "amabile", "moelleux"},
    "sweet": {"сладкое", "сладкий", "sweet", "dolce", "doux", "dulce", "suss"},
}
_SPARKLING_WORDS = {
    "игристое",
    "игристый",
    "sparkling",
    "spumante",
    "champagne",
    "шампанское",
    "brut",
    "брют",
    "sekt",
    "cava",
    "prosecco",
    "просекко",
}


# ----------------------------------------------------------------------------
# Нормализация (канон: app.pipeline.text_compare)
# ----------------------------------------------------------------------------
def normalize_text(text: str | None) -> str:
    """NFKC + normalize_compare_text (цифры-в-слове + омоглифы Vision→кириллица)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    return normalize_compare_text(t)


def tokens(text: str) -> list[str]:
    """Токены из уже нормализованного текста (unicode-буквы/цифры)."""
    return [t for t in _TOKEN_RE.findall(text) if len(t) >= 2 or t.isdigit()]


def lines_to_text(lines: Iterable[str]) -> str:
    return "\n".join(str(x).strip() for x in lines if str(x).strip())


def _char_ngrams(text: str, n: int) -> set[str]:
    s = text.replace(" ", "_")
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _safe_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def _years(text: str) -> set[str]:
    return set(_YEAR_RE.findall(text))


def _current_year() -> int:
    from datetime import date

    return date.today().year


def _as_year(value: Any) -> int | None:
    try:
        y = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if 1950 <= y <= _current_year() + 1:
        return y
    return None


def _vintage_years_from_text(text: str, foundation: int | None) -> set[int]:
    """Годы урожая из текста; foundation_year винодельни исключается."""
    out: set[int] = set()
    for raw in _years(text):
        y = _as_year(raw)
        if y is None:
            continue
        if foundation is not None and y == int(foundation):
            continue
        out.add(y)
    return out


def _numbers(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _NUM_RE.findall(text):
        v = raw.replace(",", ".")
        try:
            f = float(v)
        except ValueError:
            continue
        out.add(f"{f:g}")
    return out


def _alcohol(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _ALC_RE.findall(text):
        try:
            f = float(raw.replace(",", "."))
        except ValueError:
            continue
        if 4.0 <= f <= 25.0:
            out.add(f"{f:g}")
    return out


def _volumes_l(text: str) -> set[str]:
    out: set[str] = set()
    for num, unit in _VOL_RE.findall(text):
        try:
            f = float(num.replace(",", "."))
        except ValueError:
            continue
        if unit.lower() in ("мл", "ml"):
            f = f / 1000.0
        if 0.1 <= f <= 20.0:
            out.add(f"{f:g}")
    return out


def _detect_class(toks: Sequence[str], table: dict[str, set[str]]) -> set[str]:
    found: set[str] = set()
    tokset = set(toks)
    for cls, words in table.items():
        if tokset & words:
            found.add(cls)
    return found


def _color_from_category(category: str | None) -> str | None:
    c = normalize_text(category)
    if not c:
        return None
    for prefix, cls in _COLOR_CATEGORY_PREFIX:
        if c.startswith(prefix):
            return cls
    return None


def _fuzzy_coverage(
    src: Sequence[str], tgt: Sequence[str], thr: float = 80.0
) -> tuple[float, float, float]:
    """Для каждого токена src — лучший fuzz.ratio к токенам tgt.

    Возвращает (mean_best, max_best, доля src-токенов с best >= thr).
    """
    if not src or not tgt:
        return 0.0, 0.0, 0.0
    bests: list[float] = []
    for s in src:
        best = 0.0
        for t in tgt:
            r = fuzz.ratio(s, t)
            if r > best:
                best = r
                if best >= 100.0:
                    break
        bests.append(best)
    mean_best = sum(bests) / len(bests) / 100.0
    max_best = max(bests) / 100.0
    frac = sum(1 for b in bests if b >= thr) / len(bests)
    return mean_best, max_best, frac


# ----------------------------------------------------------------------------
# Эталон вина
# ----------------------------------------------------------------------------
@dataclass
class WineRef:
    wine_id: int
    label: str | None = None
    name: str | None = None
    winery: str | None = None
    category: str | None = None
    color: str | None = None
    grape: str | None = None
    region: str | None = None
    # Gemini-категоризация этикетки (wines.label_ocr), не CMS-поля
    label_color: str | None = None  # label_ocr.color → «белое»
    label_type: str | None = None  # label_ocr.type → «сухое»
    vintage_year: int | None = None
    foundation_year: int | None = None

    def ref_text(self) -> str:
        """Эталон = только текст этикетки (wines.label), без CMS name/winery."""
        if self.label and self.label.strip():
            return self.label.strip()
        return ""

    def cross_encoder_text(self) -> str:
        """text2 для Cross-Encoder: структурные поля + строки этикетки."""
        parts: list[str] = []
        for p in (self.name, self.winery, self.category, self.grape):
            if p and str(p).strip():
                parts.append(str(p).strip())
        lab = self.label or ""
        lab_lines = [x.strip() for x in lab.splitlines() if x.strip()]
        if lab_lines:
            parts.append(" | ".join(lab_lines))
        return " | ".join(parts)


def ocr_cross_encoder_text(ocr_text: str) -> str:
    """text1 для Cross-Encoder: строки OCR через ' | '."""
    lines = [x.strip() for x in str(ocr_text or "").splitlines() if x.strip()]
    return " | ".join(lines)


# ----------------------------------------------------------------------------
# Признаки пары
# ----------------------------------------------------------------------------
def pair_features(
    ocr_text: str,
    ref: WineRef,
    siglip_cosine: float | None,
) -> dict[str, float]:
    """Абсолютные признаки пары (без относительных по группе)."""
    o = normalize_text(ocr_text)
    r = normalize_text(ref.ref_text())
    o_lines = [normalize_text(x) for x in str(ocr_text or "").splitlines() if x.strip()]
    o_tok = tokens(o)
    r_tok = tokens(r)
    o_tokset = set(o_tok)
    r_tokset = set(r_tok)

    f: dict[str, float] = {}

    # --- embedding (SigLIP cosine кандидата в Top-N) ---
    try:
        f["siglip_cosine"] = (
            float(siglip_cosine) if siglip_cosine is not None else 0.0
        )
    except (TypeError, ValueError):
        f["siglip_cosine"] = 0.0
    if f["siglip_cosine"] != f["siglip_cosine"]:  # NaN
        f["siglip_cosine"] = 0.0

    # --- длины ---
    f["ocr_chars"] = float(len(o))
    f["ref_chars"] = float(len(r))
    f["char_length_ratio"] = _safe_ratio(len(o), len(r))
    f["ocr_lines"] = float(len(o_lines))
    f["ocr_tokens"] = float(len(o_tok))
    f["ref_tokens"] = float(len(r_tok))
    f["token_length_ratio"] = _safe_ratio(len(o_tok), len(r_tok))
    f["ref_has_label"] = 1.0 if (ref.label and ref.label.strip()) else 0.0

    # --- строковые ---
    f["lev_similarity"] = Levenshtein.normalized_similarity(o, r)
    f["jaro_winkler"] = JaroWinkler.normalized_similarity(o, r)
    f["fuzz_ratio"] = fuzz.ratio(o, r) / 100.0
    f["fuzz_partial_ratio"] = fuzz.partial_ratio(o, r) / 100.0
    f["fuzz_token_sort_ratio"] = fuzz.token_sort_ratio(o, r) / 100.0
    f["fuzz_token_set_ratio"] = fuzz.token_set_ratio(o, r) / 100.0
    f["fuzz_wratio"] = fuzz.WRatio(o, r) / 100.0
    for n in (2, 3, 4):
        f[f"char_ngram_jaccard_{n}"] = _jaccard(_char_ngrams(o, n), _char_ngrams(r, n))

    # --- токены ---
    f["token_jaccard"] = _jaccard(o_tokset, r_tokset)
    inter = len(o_tokset & r_tokset)
    f["token_overlap_ratio"] = inter / float(min(len(o_tokset), len(r_tokset))) if o_tokset and r_tokset else 0.0
    f["token_overlap_count"] = float(inter)
    m, mx, frac = _fuzzy_coverage(r_tok, o_tok)
    f["ref_tok_cov_mean"] = m
    f["ref_tok_cov_max"] = mx
    f["ref_tok_cov_frac80"] = frac
    m, mx, frac = _fuzzy_coverage(o_tok, r_tok)
    f["ocr_tok_cov_mean"] = m
    f["ocr_tok_cov_max"] = mx
    f["ocr_tok_cov_frac80"] = frac
    # длинные токены (>=5) — обычно названия/бренды
    r_long = [t for t in r_tok if len(t) >= 5 and not t.isdigit()]
    o_long = [t for t in o_tok if len(t) >= 5 and not t.isdigit()]
    m, mx, frac = _fuzzy_coverage(r_long, o_long)
    f["ref_long_tok_cov_mean"] = m
    f["ref_long_tok_cov_frac80"] = frac
    f["ref_long_tok_count"] = float(len(r_long))

    # --- годы урожая (не foundation) / числа ---
    foundation = ref.foundation_year
    oy = _vintage_years_from_text(o, foundation)
    if ref.vintage_year is not None and 1950 <= int(ref.vintage_year) <= _current_year() + 1:
        ry = {int(ref.vintage_year)}
    else:
        ry = _vintage_years_from_text(r, foundation)
    f["year_present_ocr"] = 1.0 if oy else 0.0
    f["year_present_ref"] = 1.0 if ry else 0.0
    f["year_present_both"] = 1.0 if (oy and ry) else 0.0
    f["year_exact_match"] = 1.0 if (oy & ry) else 0.0
    f["year_mismatch"] = 1.0 if (oy and ry and not (oy & ry)) else 0.0
    # совпадение свежего урожая важнее старого: 1/(1+|now-y|)
    if oy & ry:
        y = max(oy & ry)
        f["year_match_recency"] = 1.0 / (1.0 + abs(_current_year() - y))
    else:
        f["year_match_recency"] = 0.0
    all_ocr_years = _years(o)
    all_ref_years = _years(r)
    on = _numbers(o) - all_ocr_years
    rn = _numbers(r) - all_ref_years
    f["num_count_ocr"] = float(len(on))
    f["num_count_ref"] = float(len(rn))
    f["num_jaccard"] = _jaccard(on, rn)
    f["num_ref_only"] = float(len(rn - on))
    f["num_ocr_only"] = float(len(on - rn))
    oa, ra = _alcohol(o), _alcohol(r)
    f["alc_present_both"] = 1.0 if (oa and ra) else 0.0
    f["alc_match"] = 1.0 if (oa & ra) else 0.0
    f["alc_mismatch"] = 1.0 if (oa and ra and not (oa & ra)) else 0.0
    ov, rv = _volumes_l(o), _volumes_l(r)
    f["vol_present_both"] = 1.0 if (ov and rv) else 0.0
    f["vol_match"] = 1.0 if (ov & rv) else 0.0
    f["vol_mismatch"] = 1.0 if (ov and rv and not (ov & rv)) else 0.0

    # --- посимвольные (цифры / буквы отдельно) ---
    od = "".join(ch for ch in o if ch.isdigit())
    rd = "".join(ch for ch in r if ch.isdigit())
    f["digit_match_ratio"] = fuzz.ratio(od, rd) / 100.0 if (od or rd) else 0.0
    oal = "".join(ch for ch in o if ch.isalpha())
    ral = "".join(ch for ch in r if ch.isalpha())
    f["alpha_match_ratio"] = fuzz.ratio(oal, ral) / 100.0 if (oal or ral) else 0.0
    # доля кириллицы в OCR vs в эталоне (разный алфавит — сигнал несовпадения)
    def _cyr_frac(s: str) -> float:
        letters = [ch for ch in s if ch.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for ch in letters if "а" <= ch <= "я") / len(letters)

    f["cyr_frac_ocr"] = _cyr_frac(o)
    f["cyr_frac_ref"] = _cyr_frac(r)
    f["cyr_frac_diff"] = abs(f["cyr_frac_ocr"] - f["cyr_frac_ref"])

    # --- цвет / сладость / игристое: только label + label_ocr (не CMS) ---
    ocr_colors = _detect_class(o_tok, _COLOR_WORDS)
    ref_colors = _detect_class(tokens(normalize_text(ref.label_color or "")), _COLOR_WORDS)
    if not ref_colors:
        # fallback: слова цвета в тексте этикетки, не wines.category/color
        ref_colors = _detect_class(r_tok, _COLOR_WORDS)
    f["color_ocr_detected"] = 1.0 if ocr_colors else 0.0
    f["color_ref_known"] = 1.0 if ref_colors else 0.0
    f["color_match"] = 1.0 if (ocr_colors and ref_colors and ocr_colors & ref_colors) else 0.0
    f["color_mismatch"] = 1.0 if (ocr_colors and ref_colors and not (ocr_colors & ref_colors)) else 0.0

    type_tok = tokens(normalize_text(ref.label_type or ""))
    ref_all_tok = set(r_tok) | set(type_tok)
    ocr_sweet = _detect_class(o_tok, _SWEET_WORDS)
    ref_sweet = _detect_class(list(ref_all_tok), _SWEET_WORDS)
    f["sweet_ocr_detected"] = 1.0 if ocr_sweet else 0.0
    f["sweet_match"] = 1.0 if (ocr_sweet and ref_sweet and ocr_sweet & ref_sweet) else 0.0
    f["sweet_mismatch"] = 1.0 if (ocr_sweet and ref_sweet and not (ocr_sweet & ref_sweet)) else 0.0

    ocr_spark = bool(o_tokset & _SPARKLING_WORDS)
    ref_spark = bool(ref_all_tok & _SPARKLING_WORDS)
    f["sparkling_ocr"] = 1.0 if ocr_spark else 0.0
    f["sparkling_ref"] = 1.0 if ref_spark else 0.0
    f["sparkling_mismatch"] = 1.0 if (ocr_spark != ref_spark) else 0.0
    return f


def group_features(
    ocr_text: str,
    refs: Sequence[WineRef],
    cosines: Sequence[float | None] | None = None,
) -> list[dict[str, float]]:
    """Текстовые признаки + siglip_cosine (без rel_* / hard_reject)."""
    if not refs:
        return []
    if cosines is None:
        cos_list: list[float | None] = [None] * len(refs)
    else:
        cos_list = list(cosines)
        if len(cos_list) < len(refs):
            cos_list.extend([None] * (len(refs) - len(cos_list)))
    return [
        pair_features(ocr_text, ref, cos)
        for ref, cos in zip(refs, cos_list)
    ]


def _feature_names() -> list[str]:
    dummy = WineRef(
        wine_id=0,
        label="a b сухое белое 2020",
        label_color="белое",
        label_type="сухое",
        vintage_year=2020,
    )
    return list(pair_features("a b сухое белое 2020", dummy, None).keys())


FEATURE_NAMES: list[str] = _feature_names()


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "WineRef",
    "group_features",
    "lines_to_text",
    "normalize_text",
    "ocr_cross_encoder_text",
    "pair_features",
    "tokens",
    "fix_digit_letter_homoglyphs",
    "fix_latin_cyrillic_homoglyphs",
]
