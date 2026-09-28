"""Denormalized search_photos history columns for fast scan-history filters."""

from __future__ import annotations

from typing import Any

from app.db.models import SearchPhoto


def history_eval_flags(status: dict) -> tuple[int, int]:
    """Return (false_positive, false_negative) as 0/1, mutually exclusive."""
    raw = status.get("eval")
    fp = fn = 0
    if isinstance(raw, dict):
        try:
            fp = 1 if int(raw.get("false_positive") or 0) else 0
        except (TypeError, ValueError):
            fp = 0
        try:
            fn = 1 if int(raw.get("false_negative") or 0) else 0
        except (TypeError, ValueError):
            fn = 0
    if fp and fn:
        fn = 0
    return fp, fn


def eval_flags_for_manual_match(
    *,
    manual_id: int | None,
    matched_id: int | None,
) -> tuple[int, int] | None:
    """FP/FN from «Это вино» vs pipeline match.

    - no match + manual → FN
    - match ≠ manual → FP
    - match == manual → clear both (TP)
    - manual cleared → None (do not auto-change eval)
    """
    if manual_id is None:
        return None
    try:
        man = int(manual_id)
    except (TypeError, ValueError):
        return None
    if matched_id is None:
        return 0, 1
    try:
        mid = int(matched_id)
    except (TypeError, ValueError):
        return 0, 1
    if mid != man:
        return 1, 0
    return 0, 0


def apply_status_eval_flags(
    status: dict | None,
    fp: int,
    fn: int,
) -> dict:
    """Write mutually exclusive eval flags into status JSON."""
    st = dict(status or {})
    fp_i = 1 if int(fp or 0) else 0
    fn_i = 1 if int(fn or 0) else 0
    if fp_i and fn_i:
        fn_i = 0
    st["eval"] = {"false_positive": fp_i, "false_negative": fn_i}
    return st


OFFICIAL_MATCH_SOURCES = frozenset(
    {
        "final_score",
        "final_score2",
        "soft_tfidf",
        "fin1",
        "fin2",
        "xgb",
        "xgb_fin",
        "crenc",
        "crenc_fin",
        "crenc_srv",
        "empty_ocr_cosine",
        "eval_cosine",
        "eval_top1",
        "xgb_dead_fin2",
        "xgb_dead_cosine",
    }
)


def _is_official_match_source(source: str) -> bool:
    s = str(source or "").strip().lower()
    return s in OFFICIAL_MATCH_SOURCES or s.startswith("xgb_dead_")


def history_resolve_matched_wine_id(status: dict) -> int | None:
    """Official matched wine from pipeline decision only.

    Do **not** fall back to ocr_wine_id best_final / best_wine_id — that is
    only a ranking hint and must not look like a found match in history when
    text_match_decision.band is none.
    """
    matched = status.get("matched_wine") or {}
    source = str(matched.get("source") or "").strip().lower()

    def _as_int(raw: object) -> int | None:
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    mid = _as_int(status.get("matched_wine_id"))
    if mid is not None:
        return mid

    # Legacy statuses: card present with official source, id not mirrored.
    if _is_official_match_source(source):
        mid = _as_int(matched.get("id"))
        if mid is not None:
            return mid
    return None


def _as_float(raw: object) -> float | None:
    try:
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def history_extract_fields(status: dict) -> dict[str, Any]:
    """Compute denormalized history fields from status JSON."""
    matched_raw = status.get("matched_wine") or {}
    source = str(matched_raw.get("source") or "").strip().lower()
    mid = history_resolve_matched_wine_id(status)
    fp, fn = history_eval_flags(status)

    matched: dict = {}
    if mid is not None and _is_official_match_source(source):
        try:
            if int(matched_raw.get("id") or 0) == int(mid):
                matched = matched_raw
        except (TypeError, ValueError):
            matched = {}
    elif mid is not None and isinstance(matched_raw, dict):
        try:
            if int(matched_raw.get("id") or 0) == int(mid):
                matched = matched_raw
        except (TypeError, ValueError):
            matched = {}

    confidence = None
    if matched:
        confidence = status.get("matched_wine_confidence")
        if confidence is None:
            confidence = matched.get("confidence") or matched.get("final_score")
    # No ocr_wine_id best_final fallback: that is not an official match.

    slug = matched.get("slug") if matched else None
    if not slug and isinstance(matched_raw, dict):
        try:
            if mid is not None and int(matched_raw.get("id") or 0) == int(mid):
                slug = matched_raw.get("slug")
        except (TypeError, ValueError):
            pass

    timings = status.get("timings_ms") or {}
    return {
        "hist_wine_id": int(mid) if mid is not None else None,
        "hist_wine_confidence": _as_float(confidence),
        "hist_wine_slug": str(slug).strip() if slug else None,
        "hist_total_ms": _as_float(timings.get("total")),
        "hist_fp": int(fp),
        "hist_fn": int(fn),
    }


def sync_search_photo_history_columns(
    row: SearchPhoto,
    status: dict | None = None,
    *,
    wine_slug: str | None = None,
) -> None:
    """Write denormalized history filter columns onto the ORM row (no commit)."""
    st = status if status is not None else dict(row.status or {})
    fields = history_extract_fields(st if isinstance(st, dict) else {})
    if wine_slug and not fields["hist_wine_slug"]:
        fields["hist_wine_slug"] = str(wine_slug).strip() or None
    row.hist_wine_id = fields["hist_wine_id"]
    row.hist_wine_confidence = fields["hist_wine_confidence"]
    row.hist_wine_slug = fields["hist_wine_slug"]
    row.hist_total_ms = fields["hist_total_ms"]
    row.hist_fp = fields["hist_fp"]
    row.hist_fn = fields["hist_fn"]
