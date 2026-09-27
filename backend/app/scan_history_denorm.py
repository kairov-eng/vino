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
    }
)


def history_resolve_matched_wine_id(status: dict) -> int | None:
    """Official matched wine from pipeline decision (fin1/fin2/xgb/crenc/…)."""
    help_ = (status.get("steps") or {}).get("ocr_wine_id") or {}
    matched = status.get("matched_wine") or {}
    source = str(matched.get("source") or "").strip().lower()

    def _as_int(raw: object) -> int | None:
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    # Prefer status.matched_wine_id when pipeline wrote an official match card.
    mid = _as_int(status.get("matched_wine_id"))
    if mid is not None and (
        source in OFFICIAL_MATCH_SOURCES
        or (isinstance(matched, dict) and matched.get("id") is not None)
    ):
        return mid

    if source in OFFICIAL_MATCH_SOURCES:
        mid = _as_int(matched.get("id"))
        if mid is not None:
            return mid

    # fin1 then fin2 from ocr_wine_id help (ignore siglip2/dinov3 fallback)
    for key in ("best_wine_id", "best_wine_id2"):
        mid = _as_int(help_.get(key))
        if mid is not None:
            return mid
    for key in ("best_final", "best_final2"):
        row = help_.get(key) or {}
        if isinstance(row, dict):
            mid = _as_int(row.get("id"))
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
    help_ = (status.get("steps") or {}).get("ocr_wine_id") or {}
    matched_raw = status.get("matched_wine") or {}
    source = str(matched_raw.get("source") or "").strip().lower()
    mid = history_resolve_matched_wine_id(status)
    fp, fn = history_eval_flags(status)

    matched: dict = {}
    if mid is not None and source in OFFICIAL_MATCH_SOURCES:
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
    if confidence is None and mid is not None:
        bf = help_.get("best_final") or {}
        bf2 = help_.get("best_final2") or {}
        if isinstance(bf, dict) and int(bf.get("id") or 0) == int(mid):
            confidence = bf.get("final_score")
            if not matched:
                matched = {
                    "name": bf.get("name"),
                    "slug": bf.get("slug"),
                }
        elif isinstance(bf2, dict) and int(bf2.get("id") or 0) == int(mid):
            confidence = bf2.get("final_score2")
            if not matched:
                matched = {
                    "name": bf2.get("name"),
                    "slug": bf2.get("slug"),
                }

    slug = matched.get("slug") if matched else None
    if not slug and isinstance(matched_raw, dict):
        try:
            if mid is not None and int(matched_raw.get("id") or 0) == int(mid):
                slug = matched_raw.get("slug")
        except (TypeError, ValueError):
            pass
    if not slug and mid is not None:
        for key in ("best_final", "best_final2"):
            row = help_.get(key) or {}
            if isinstance(row, dict) and int(row.get("id") or 0) == int(mid):
                slug = row.get("slug")
                if slug:
                    break

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
