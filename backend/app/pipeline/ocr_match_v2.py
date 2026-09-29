"""Soft TF-IDF text match (algorithm 2): asymmetric cover_Q / cover_R + IDF.

Query Q = OCR (noisy / cropped). Reference R = catalog label (+ name/winery).
Only used for Google Vision and Yandex OCR channels.

TextScore2 ∈ [0,1] ≈ 0.70·cover_Q + 0.15·cover_R + 0.15·partial_ratio
FinalScore2 = w_text·TextScore2 + w_cos·Cosine  (same weights as FinalScore v1)
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Wine
from app.pipeline.text_compare import normalize_compare_text

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
except ImportError:  # pragma: no cover
    fuzz = None  # type: ignore
    JaroWinkler = None  # type: ignore

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

_YEAR_OCR_RE = re.compile(
    r"\b(?:19|20)[\dOoIl]{2}\b",
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
    "vol",
}

# Homoglyph / transliteration map (lowercase Cyr → latin-ish keys)
_FOLD_MAP = {
    "а": "a",
    "б": "b",
    "в": "b",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "h",
    "о": "o",
    "п": "p",
    "р": "p",
    "с": "c",
    "т": "t",
    "у": "y",
    "ф": "f",
    "х": "x",
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

# Spaced letter collapse handled per-line in _collapse_spaced_letters

# mode: "legacy" (incl. name) | "fast_no_name" (winery/grape/category/type/label)
_IDF_CACHE: dict[str, Any] = {
    "n_docs": -1,
    "mode": None,
    "idf": {},
    "default_idf": 0.0,
    "built_ms": 0.0,
}

# Soft TF-IDF blend into TextScore2
# cover_R intentionally weak / unused in main blend: shorter catalog labels
# (empty «Б Ю Р Н Ь Е») must not beat full labels that contain the same Q tokens.
_W_COVER_Q = 0.80
_W_PARTIAL = 0.20
_W_UNIQ_HIT = 0.08  # bonus only when Q hits pool-unique tokens (e.g. Classic in OCR)

SOFT_CHANNELS = frozenset({"google_vision", "yandex"})

# Veto FinalScore(v1) winner when Soft TF-IDF says OCR barely covers catalog text.
# Scan 249: v1 boost→92 / cos high, but cover_Q≈0.15 → reject match.
SOFT_VETO_COVER_Q = 0.40
SOFT_VETO_SCORE2 = 0.35


def fold_token(token: str) -> str:
    """Lowercase + Cyr→Lat fold for soft matching keys."""
    t = normalize_compare_text(token)
    if not t:
        return ""
    out: list[str] = []
    for ch in t:
        out.append(_FOLD_MAP.get(ch, ch))
    s = "".join(out)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _collapse_spaced_letters(text: str) -> str:
    """Collapse 'Б Ю Р Н Ь Е' → 'БЮРНЬЕ' per line (do not cross newlines)."""

    def _join_line(line: str) -> str:
        def _join(m: re.Match[str]) -> str:
            return re.sub(r"[ \t]+", "", m.group(0))

        # Only spaces/tabs between single letters — not newlines
        pat = re.compile(
            r"(?:(?<=[ \t])|^)(?:[A-Za-zА-Яа-яЁё][ \t]+){2,}[A-Za-zА-Яа-яЁё](?=[ \t]|$)",
            re.UNICODE,
        )
        return pat.sub(_join, line)

    return "\n".join(_join_line(ln) for ln in (text or "").splitlines())


def soft_normalize(text: str) -> str:
    """Noise strip + spaced-letter collapse + lowercase (pre-tokenize)."""
    t = text or ""
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r", "\n").replace("ё", "е").replace("Ё", "е")
    t = _collapse_spaced_letters(t)

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
    t = re.sub(r"[^\w\s\-А-Яа-яA-Za-z0-9]+", " ", t, flags=re.UNICODE)
    return normalize_compare_text(t)


def soft_tokenize(text: str) -> list[str]:
    """Set of folded tokens; keep years; drop short noise / stopwords."""
    norm = soft_normalize(text)
    if not norm:
        return []
    out: list[str] = []
    seen_local: set[str] = set()
    for raw in norm.split():
        is_year = bool(re.fullmatch(r"(?:19|20)\d{2}", raw))
        if not is_year and len(raw) < 3:
            continue
        if raw in _STOP_TOKENS:
            continue
        if raw.isdigit() and not is_year:
            continue
        folded = fold_token(raw) if not is_year else raw
        if not folded:
            continue
        if not is_year and len(folded) < 3:
            continue
        if folded in seen_local:
            continue
        seen_local.add(folded)
        out.append(folded)
    return out


def tok_sim(a: str, b: str) -> float:
    """Jaro–Winkler with length-dependent accept threshold (else 0)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if JaroWinkler is None:
        if fuzz is None:
            return 1.0 if a == b else 0.0
        s = float(fuzz.ratio(a, b)) / 100.0
    else:
        s = float(JaroWinkler.normalized_similarity(a, b))
    L = min(len(a), len(b))
    thr = 0.92 if L <= 4 else 0.85 if L <= 7 else 0.80
    return s if s >= thr else 0.0


def cover(
    q_tokens: list[str],
    r_tokens: list[str],
    idf: dict[str, float],
    *,
    default_idf: float,
    conf: dict[str, float] | None = None,
) -> float:
    """Weighted soft coverage of q by r (asymmetric)."""
    if not q_tokens:
        return 0.0
    num = den = 0.0
    for t in q_tokens:
        w = float(idf.get(t, default_idf))
        if conf:
            w *= float(conf.get(t, 1.0))
        s = max((tok_sim(t, r) for r in r_tokens), default=0.0)
        num += w * s
        den += w
    return (num / den) if den else 0.0


def partial_ratio_01(q_text: str, r_text: str) -> float:
    if not q_text or not r_text or fuzz is None:
        return 0.0
    q = soft_normalize(q_text).replace(" ", "")
    r = soft_normalize(r_text).replace(" ", "")
    if not q or not r:
        return 0.0
    # shorter as query for partial (Q should sit inside R)
    if len(q) > len(r):
        q, r = r, q
    return float(fuzz.partial_ratio(q, r)) / 100.0


def char_ngram_dice(a: str, b: str, n: int = 3) -> float:
    """Dice on character n-grams — glue/split OCR insurance."""
    aa = soft_normalize(a).replace(" ", "")
    bb = soft_normalize(b).replace(" ", "")
    if len(aa) < n or len(bb) < n:
        return 0.0
    ca = Counter(aa[i : i + n] for i in range(len(aa) - n + 1))
    cb = Counter(bb[i : i + n] for i in range(len(bb) - n + 1))
    inter = sum((ca & cb).values())
    return (2.0 * inter) / (sum(ca.values()) + sum(cb.values()))


def wine_reference_text(wine: Wine, *, exclude_name: bool = False) -> str:
    """Текст R для Soft TF-IDF.

    exclude_name=True (fast_text_match): без CMS name — название часто
    искажается OCR; оставляем winery / label / grape / category / color
    (type сахаристости обычно в label). Region тоже не берём — не в
    exclusive-сравнении.
    """
    if exclude_name:
        parts = [
            wine.winery or "",
            wine.label or "",
            wine.grape_variety or "",
            wine.category or "",
            wine.color or "",
        ]
    else:
        parts = [
            wine.winery or "",
            wine.name or "",
            wine.label or "",
            wine.grape_variety or "",
            wine.region or "",
            wine.category or "",
            wine.color or "",
        ]
    return "\n".join(p for p in parts if str(p).strip())


def build_catalog_idf(
    db: Session,
    *,
    exclude_name: bool = False,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """IDF по каталогу для Soft TF-IDF. Кеш по (n_docs, mode).

    Важно: это не сравнение со всеми винами — только веса редкости токенов.
    Скоринг Soft идёт только по visual-кандидатам (top-N).
    """
    global _IDF_CACHE
    mode = "fast_no_name" if exclude_name else "legacy"
    n_docs = int(db.scalar(select(func.count()).select_from(Wine)) or 0)
    if (
        _IDF_CACHE.get("n_docs") == n_docs
        and _IDF_CACHE.get("mode") == mode
        and isinstance(_IDF_CACHE.get("idf"), dict)
        and _IDF_CACHE["idf"]
    ):
        return (
            _IDF_CACHE["idf"],
            float(_IDF_CACHE.get("default_idf") or math.log(max(n_docs, 2))),
            {
                "cached": True,
                "n_docs": n_docs,
                "mode": mode,
                "ms": _IDF_CACHE.get("built_ms", 0),
            },
        )

    t0 = time.perf_counter()
    df: Counter[str] = Counter()
    if exclude_name:
        rows = db.execute(
            select(
                Wine.winery,
                Wine.label,
                Wine.grape_variety,
                Wine.category,
                Wine.color,
            )
        ).all()
    else:
        rows = db.execute(
            select(
                Wine.winery,
                Wine.name,
                Wine.label,
                Wine.grape_variety,
                Wine.region,
                Wine.category,
                Wine.color,
            )
        ).all()
    for row in rows:
        blob = "\n".join(str(x or "") for x in row)
        toks = set(soft_tokenize(blob))
        for t in toks:
            df[t] += 1
    idf: dict[str, float] = {}
    n = max(n_docs, 1)
    for t, d in df.items():
        # smoothed IDF
        idf[t] = math.log((1.0 + n) / (1.0 + d)) + 1.0
    default_idf = math.log(max(n, 2))
    ms = (time.perf_counter() - t0) * 1000.0
    _IDF_CACHE = {
        "n_docs": n_docs,
        "mode": mode,
        "idf": idf,
        "default_idf": default_idf,
        "built_ms": round(ms, 1),
    }
    return (
        idf,
        default_idf,
        {"cached": False, "n_docs": n_docs, "mode": mode, "ms": round(ms, 1)},
    )


def score_soft_tfidf(
    query_text: str,
    ref_text: str,
    idf: dict[str, float],
    default_idf: float,
    *,
    pool_unique: set[str] | None = None,
) -> dict[str, Any]:
    """Compute TextScore2 and diagnostics for one (Q, R) pair."""
    q_toks = soft_tokenize(query_text)
    r_toks = soft_tokenize(ref_text)
    cq = cover(q_toks, r_toks, idf, default_idf=default_idf)
    cr = cover(r_toks, q_toks, idf, default_idf=default_idf)
    pr = partial_ratio_01(query_text, ref_text)
    ng = char_ngram_dice(query_text, ref_text, 3)

    # Unique-in-pool recall: of R tokens that distinguish this candidate, how many in Q
    uniq_recall = 0.0
    if pool_unique:
        uniq_r = [t for t in r_toks if t in pool_unique]
        if uniq_r:
            hits = sum(
                1.0
                for t in uniq_r
                if max((tok_sim(t, q) for q in q_toks), default=0.0) > 0
            )
            uniq_recall = hits / len(uniq_r)

    # Blend: cover_Q dominant; n-gram as light insurance inside partial slot.
    # cover_R kept in diagnostics only (asymmetric: crop-tolerant, not for ranking
    # empty/short labels above full ones).
    partial_blend = max(pr, 0.5 * ng)
    score = (
        _W_COVER_Q * cq
        + _W_PARTIAL * partial_blend
        + _W_UNIQ_HIT * uniq_recall
    )
    score = max(0.0, min(1.0, score))

    matched_idf = 0.0
    for t in q_toks:
        s = max((tok_sim(t, r) for r in r_toks), default=0.0)
        if s > 0:
            matched_idf += float(idf.get(t, default_idf)) * s

    return {
        "score_01": round(score, 6),
        "cover_q": round(cq, 6),
        "cover_r": round(cr, 6),
        "partial_ratio": round(pr, 6),
        "ngram3": round(ng, 6),
        "uniq_recall": round(uniq_recall, 6),
        "matched_idf": round(matched_idf, 4),
        "q_tokens": q_toks,
        "r_tokens": r_toks,
        "q_token_count": len(q_toks),
        "r_token_count": len(r_toks),
    }


def pool_unique_tokens(ref_token_lists: dict[int, list[str]]) -> dict[int, set[str]]:
    """Tokens that appear in exactly one candidate's reference (e.g. Classic)."""
    df: Counter[str] = Counter()
    for toks in ref_token_lists.values():
        for t in set(toks):
            df[t] += 1
    out: dict[int, set[str]] = {}
    for wid, toks in ref_token_lists.items():
        out[wid] = {t for t in set(toks) if df[t] == 1}
    return out


def score_channel_soft(
    *,
    query_text: str,
    wines: list[Wine],
    cosine_by_id: dict[int, float],
    idf: dict[str, float],
    default_idf: float,
    fweights: dict[str, float] | None = None,
    exclusive_reject_ids: set[int] | frozenset[int] | None = None,
    exclude_name: bool = False,
) -> list[dict[str, Any]]:
    """Score all visual candidates; return rows with score2 / final_score2."""
    # Late import to avoid circular dependency with ocr_match
    from app.pipeline.ocr_match import compute_final_score, final_score_weights

    fw = fweights or final_score_weights()
    exclusive_ids = {int(x) for x in (exclusive_reject_ids or set())}
    ref_toks: dict[int, list[str]] = {}
    refs: dict[int, str] = {}
    for w in wines:
        ref = wine_reference_text(w, exclude_name=exclude_name)
        refs[int(w.id)] = ref
        ref_toks[int(w.id)] = soft_tokenize(ref)
    uniq = pool_unique_tokens(ref_toks)

    scored: list[dict[str, Any]] = []
    for w in wines:
        wid = int(w.id)
        detail = score_soft_tfidf(
            query_text,
            refs[wid],
            idf,
            default_idf,
            pool_unique=uniq.get(wid),
        )
        cos = cosine_by_id.get(wid)
        text01 = float(detail["score_01"])
        # Too little informative query text → don't trust high cover of noise
        if detail["q_token_count"] < 2:
            text01 *= 0.5
        excl = wid in exclusive_ids
        if excl:
            final = {
                "final_score": 0.0,
                "text_score_01": 0.0,
                "cosine": None if cos is None else round(float(cos), 6),
                "weights": fw,
                "hard_mismatch": True,
                "mismatch_fields": ["exclusive_lexicon"],
            }
            text01 = 0.0
        else:
            final = compute_final_score(
                text_score_01=text01,
                cosine=cos,
                weights=fw,
            )
        scored.append(
            {
                "id": wid,
                "name": w.name,
                "winery": w.winery,
                "score2": round(text01 * 100.0, 2),
                "score2_01": round(text01, 6),
                "final_score2": final["final_score"],
                "cosine": None if cos is None else round(float(cos), 6),
                "exclusive_rejected": excl,
                "hard_mismatch": excl,
                "soft_tfidf": {
                    k: detail[k]
                    for k in (
                        "cover_q",
                        "cover_r",
                        "partial_ratio",
                        "ngram3",
                        "uniq_recall",
                        "matched_idf",
                        "q_token_count",
                        "r_token_count",
                    )
                },
            }
        )
    scored.sort(
        key=lambda x: (
            -float(x.get("final_score2") or 0),
            -float(x.get("score2") or 0),
            -float((x.get("soft_tfidf") or {}).get("cover_q") or 0),
            # Prefer richer references when cover_Q ties (more r tokens matched via partial)
            -float((x.get("soft_tfidf") or {}).get("r_token_count") or 0),
            x["id"],
        )
    )
    return scored
