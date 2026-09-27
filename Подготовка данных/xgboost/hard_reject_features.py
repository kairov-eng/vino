"""Признаки причин hard reject для кандидата из search_photos.status.

Признаки привязаны к (original_scan_id, wine_id, ocr_engine). Сначала берётся
engine-specific `ocr_wine_id.per_variant[engine].visual_support`, затем fallback
на общий `ocr_wine_id.final_ranked`. HSV hard reject не зависит от OCR engine.
"""
from __future__ import annotations

from typing import Any

HARD_REJECT_FEATURE_VERSION = "1.0.0"

HARD_REJECT_FIELDS = (
    "name",
    "winery",
    "grape",
    "color",
    "type",
    "exclusive_lexicon",
    "label_lines",
)

HARD_REJECT_FEATURE_NAMES = [
    "hard_reject_detail_available",
    "hard_reject_any",
    "hard_reject_count",
    *(f"hard_reject_{field}" for field in HARD_REJECT_FIELDS),
    "hard_reject_label_any",
    "hard_reject_label_majority_unmatched",
    "hard_reject_label_weak_structure_low_cosine",
    "hard_reject_hsv",
    "hard_reject_unknown_reason",
]


def _candidate_map(rows: Any) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        try:
            out[int(row["id"])] = row
        except (TypeError, ValueError):
            continue
    return out


def build_reject_index(
    status: dict[str, Any],
) -> tuple[dict[str, dict[int, dict[str, Any]]], set[int]]:
    """Индекс `{engine|_final: {wine_id: detail}}` и HSV hard-rejected ids."""
    help_step = ((status.get("steps") or {}).get("ocr_wine_id") or {})
    index: dict[str, dict[int, dict[str, Any]]] = {
        "_final": _candidate_map(help_step.get("final_ranked") or [])
    }
    for engine, entry in (help_step.get("per_variant") or {}).items():
        if not isinstance(entry, dict):
            continue
        rows = entry.get("visual_support") or entry.get("top_matches") or []
        index[str(engine)] = _candidate_map(rows)

    hsv = ((status.get("steps") or {}).get("hsv") or {})
    hard_ids: set[int] = set()
    for raw in hsv.get("hard_rejected_ids") or []:
        try:
            hard_ids.add(int(raw))
        except (TypeError, ValueError):
            pass
    return index, hard_ids


def hard_reject_features(
    index: dict[str, dict[int, dict[str, Any]]],
    hsv_hard_ids: set[int],
    *,
    wine_id: int,
    ocr_engine: str,
) -> dict[str, float]:
    features = {name: 0.0 for name in HARD_REJECT_FEATURE_NAMES}
    engine = str(ocr_engine or "").removeprefix("synthetic_")
    detail = (index.get(engine) or {}).get(int(wine_id))
    if detail is None:
        detail = (index.get("_final") or {}).get(int(wine_id))
    if detail is not None:
        features["hard_reject_detail_available"] = 1.0
        mismatch_fields = {
            str(value).strip().lower()
            for value in (detail.get("mismatch_fields") or [])
            if str(value).strip()
        }
        is_hard = bool(detail.get("hard_mismatch"))
        features["hard_reject_any"] = float(is_hard)
        for field in HARD_REJECT_FIELDS:
            features[f"hard_reject_{field}"] = float(field in mismatch_fields)
        features["hard_reject_count"] = float(len(mismatch_fields))

        line_info = detail.get("label_lines") or {}
        line_reject = bool(line_info.get("reject"))
        reason = str(line_info.get("reject_reason") or "").strip().lower()
        features["hard_reject_label_any"] = float(line_reject)
        features["hard_reject_label_majority_unmatched"] = float(
            reason == "majority_unmatched_lines"
        )
        features["hard_reject_label_weak_structure_low_cosine"] = float(
            reason == "weak_structure_low_cosine"
        )
        known = set(HARD_REJECT_FIELDS)
        features["hard_reject_unknown_reason"] = float(
            is_hard and not (mismatch_fields & known)
        )

    features["hard_reject_hsv"] = float(int(wine_id) in hsv_hard_ids)
    # HSV — самостоятельный абсолютный hard reject.
    if features["hard_reject_hsv"]:
        features["hard_reject_any"] = 1.0
    return features


__all__ = [
    "HARD_REJECT_FEATURE_NAMES",
    "HARD_REJECT_FEATURE_VERSION",
    "build_reject_index",
    "hard_reject_features",
]
