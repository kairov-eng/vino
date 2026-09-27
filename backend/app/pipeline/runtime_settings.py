"""Persistent pipeline settings (OCR engines/preprocess + TextScore weights + search flags).

Source of truth for UI toggles: backend/pipeline_settings.json.
Applied to in-memory app_config on get_settings()/save_settings()/app startup
(no .env writes). Secrets and endpoint URLs stay in .env only.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.db import config as app_config

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "pipeline_settings.json"

PREPROCESS_ALL = ("A", "B", "C", "D")
ENGINE_ALL = ("rapid", "easy", "surya", "tess")
# EasyOCR / Surya / Tesseract в этом пайплайне не используются.
ENGINE_DISABLED = frozenset({"easy", "surya", "tess"})
# Порядок: название → винодельня → цвет → тип → купаж → год → …
WEIGHT_KEYS = (
    "wine",
    "producer",
    "color",
    "type",
    "grape",
    "vintage",
    "region",
    "other",
    "brand",
)
FINAL_WEIGHT_KEYS = ("text", "cosine")

PREPROCESS_META = {
    "A": {"id": "A", "name": "original", "title": "Original"},
    "B": {"id": "B", "name": "upscale_clahe", "title": "Upscale + CLAHE"},
    "C": {"id": "C", "name": "gray_sharpen", "title": "Gray + Sharpen"},
    "D": {"id": "D", "name": "gray_adaptive_threshold", "title": "Adaptive Threshold"},
}

ENGINE_META = {
    "rapid": {"id": "rapid", "name": "rapidocr", "title": "RapidOCR"},
    "easy": {"id": "easy", "name": "easyocr", "title": "EasyOCR"},
    "surya": {"id": "surya", "name": "surya", "title": "Surya"},
    "tess": {"id": "tess", "name": "tesseract", "title": "Tesseract"},
}

WEIGHT_META = {
    "wine": {"key": "wine", "label": "Название вина", "env": "TEXT_WEIGHT_WINE"},
    "producer": {
        "key": "producer",
        "label": "Винодельня",
        "env": "TEXT_WEIGHT_PRODUCER",
    },
    "color": {"key": "color", "label": "Цвет", "env": "TEXT_WEIGHT_COLOR"},
    "type": {"key": "type", "label": "Тип (сухое…)", "env": "TEXT_WEIGHT_TYPE"},
    "grape": {"key": "grape", "label": "Купаж / сорта", "env": "TEXT_WEIGHT_GRAPE"},
    "vintage": {"key": "vintage", "label": "Год", "env": "TEXT_WEIGHT_VINTAGE"},
    "region": {"key": "region", "label": "Регион", "env": "TEXT_WEIGHT_REGION"},
    "other": {"key": "other", "label": "Прочее", "env": "TEXT_WEIGHT_OTHER"},
    "brand": {"key": "brand", "label": "Brand overlap", "env": "TEXT_WEIGHT_BRAND"},
}

FINAL_WEIGHT_META = {
    "text": {
        "key": "text",
        "label": "OCR (TextScore)",
        "env": "FINAL_SCORE_W_TEXT",
    },
    "cosine": {
        "key": "cosine",
        "label": "Embedding (cosine)",
        "env": "FINAL_SCORE_W_COS",
    },
}

_lock = threading.Lock()


def _defaults() -> dict[str, Any]:
    preprocess = getattr(app_config, "OCR_PREPROCESS", None)
    if not preprocess:
        preprocess = list(PREPROCESS_ALL)
    raw_engines = getattr(app_config, "OCR_ENGINES", None)
    if raw_engines is None:
        engines = list(ENGINE_ALL)
    else:
        eng_set = {str(x).lower() for x in raw_engines}
        engines = [e for e in ENGINE_ALL if e in eng_set]
    engines = [e for e in engines if e not in ENGINE_DISABLED]
    weights = {
        "wine": float(getattr(app_config, "TEXT_WEIGHT_WINE", 0.40)),
        "producer": float(getattr(app_config, "TEXT_WEIGHT_PRODUCER", 0.30)),
        "color": float(getattr(app_config, "TEXT_WEIGHT_COLOR", 0.14)),
        "type": float(getattr(app_config, "TEXT_WEIGHT_TYPE", 0.08)),
        "grape": float(getattr(app_config, "TEXT_WEIGHT_GRAPE", 0.12)),
        "vintage": float(getattr(app_config, "TEXT_WEIGHT_VINTAGE", 0.15)),
        "region": float(getattr(app_config, "TEXT_WEIGHT_REGION", 0.10)),
        "other": float(getattr(app_config, "TEXT_WEIGHT_OTHER", 0.05)),
        "brand": float(getattr(app_config, "TEXT_WEIGHT_BRAND", 0.12)),
    }
    final_weights = {
        "text": float(getattr(app_config, "FINAL_SCORE_W_TEXT", 0.45)),
        "cosine": float(getattr(app_config, "FINAL_SCORE_W_COS", 0.55)),
    }
    return {
        "ocr_preprocess": [p for p in PREPROCESS_ALL if p in set(preprocess)],
        "ocr_engines": engines,
        "text_weights": weights,
        "final_weights": final_weights,
        "use_dinov3": bool(getattr(app_config, "USE_DINOV3", True)),
        "geometry_siglip2": bool(getattr(app_config, "GEOMETRY_SIGLIP2", True)),
        "geometry_dinov3": bool(getattr(app_config, "GEOMETRY_DINOV3", True)),
        "use_gemini_ocr": bool(getattr(app_config, "USE_GEMINI_OCR", True)),
        "use_openai_ocr": bool(getattr(app_config, "USE_OPENAI_OCR", True)),
        "use_deepseek_ocr": bool(getattr(app_config, "USE_DEEPSEEK_OCR", True)),
        "use_qwen_ocr": bool(getattr(app_config, "USE_QWEN_OCR", False)),
        "use_yandex_ocr": bool(getattr(app_config, "USE_YANDEX_OCR", True)),
        "use_google_vision_ocr": bool(
            getattr(app_config, "USE_GOOGLE_VISION_OCR", True)
        ),
        "yolo_fallback_full_image": bool(
            getattr(app_config, "YOLO_FALLBACK_FULL_IMAGE", True)
        ),
        "use_yolo": bool(getattr(app_config, "USE_YOLO", True)),
        "yolo_variant": _normalize_yolo_variant(
            getattr(app_config, "YOLO_VARIANT", "label")
        ),
        "bottle_min_conf": 0.60,
        "use_bottle_orient": True,
        "bottle_orient_min_deg": 8.0,
        "label_detect_openai": bool(
            getattr(app_config, "LABEL_DETECT_OPENAI", True)
        ),
        "label_detect_gemini": bool(
            getattr(app_config, "LABEL_DETECT_GEMINI", True)
        ),
        "embedding_device": _normalize_embedding_device(
            getattr(app_config, "EMBEDDING_DEVICE", "cpu")
        ),
        "embed_cpu_use_cache": bool(
            getattr(app_config, "EMBED_CPU_USE_CACHE", False)
        ),
        "normalize_max_side": _normalize_max_side(
            getattr(app_config, "NORMALIZE_MAX_SIDE", 1024)
        ),
        "hf_start_timeout_sec": _normalize_hf_start_timeout(
            getattr(app_config, "HF_START_TIMEOUT_SEC", 15)
        ),
        "text_match_methods": list(TEXT_MATCH_DEFAULT),
        "final_score_method": "fin1",
        "text_match_thresholds": _default_text_match_thresholds(),
        # Empty OCR: match only if max cosine among candidates ≥ this.
        "empty_ocr_cosine_threshold": 0.86,
        # Dead-XGB: if max xgb_score among exclusive survivors < τ →
        # Soft TF-IDF (fin2 always, even if disabled in methods) or cosine.
        "xgb_dead_max": 0.15,
        # Master switch: when False, skip HSV compute + ignore filter/hard gates.
        "compute_hsv": False,
        "use_hsv_filter": False,
        "hsv_bhattacharyya_max": 0.50,
        "hsv_ignore_high_cosine": False,
        "hsv_ignore_cosine_min": 0.85,
        "use_hsv_hard_reject": False,
        "hsv_hard_reject_max": 0.90,
        # Dominant-color CIEDE2000 (ColorDelta) on candidate cards.
        "compute_color_delta": True,
        # Reuse past search: nearest siglip2 search_photo embedding ≥0.5
        # → inject winner (search_photos_id=prev) + catalog top-19 (id=0);
        # use that search's GV OCR (skip live GV when text present).
        "reuse_previous_searches": False,
        # Admin UI: show technical search details (candidates, OCR, scores).
        # Non-admins always see the simplified scanner regardless of this flag.
        "show_search_details": True,
        "final_ocr": "auto",
        "exclusive_use_translit": True,
        "exclusive_match_spaced": True,
    }


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_yolo_variant(value: Any) -> str:
    raw = str(value or "label").strip().lower().replace("-", "_")
    if raw in {"bottle_label", "bottle+label", "bottlelabel", "bl"}:
        return "bottle_label"
    return "label"


NORMALIZE_MAX_SIDE_OPTIONS = (1280, 1024, 800)

TEXT_MATCH_ALL = ("fin1", "fin2", "xgb", "crenc", "crenc_srv", "openai_txt")
# openai_txt off by default (API cost); enable via settings checkbox
TEXT_MATCH_DEFAULT = ("fin1", "fin2", "xgb", "crenc", "crenc_srv")
TEXT_MATCH_META: dict[str, dict[str, str]] = {
    "fin1": {"id": "fin1", "label": "fin1 (TextScore)"},
    "fin2": {"id": "fin2", "label": "fin2 (Soft TF-IDF)"},
    "xgb": {"id": "xgb", "label": "XGBoost"},
    "crenc": {"id": "crenc", "label": "Cross Encoder (local)"},
    "crenc_srv": {"id": "crenc_srv", "label": "Cross Encoder (server)"},
    "openai_txt": {"id": "openai_txt", "label": "OpenAI сравнение текста"},
}

FINAL_OCR_ALL = (
    "auto",
    "google_vision",
    "gemini",
    "openai",
    "yandex",
    "deepseek",
    "qwen",
)
FINAL_OCR_META: dict[str, dict[str, str]] = {
    "auto": {"id": "auto", "label": "Auto (приоритет)"},
    "google_vision": {"id": "google_vision", "label": "Google Vision"},
    "gemini": {"id": "gemini", "label": "Gemini"},
    "openai": {"id": "openai", "label": "OpenAI"},
    "yandex": {"id": "yandex", "label": "Yandex"},
    "deepseek": {"id": "deepseek", "label": "DeepSeek"},
    "qwen": {"id": "qwen", "label": "Qwen"},
}

# Defaults: score > match → hit; score < similar → miss; else similar-band
_DEFAULT_MATCH_THR = 0.55
_DEFAULT_SIMILAR_THR = 0.25


def _clamp01(value: Any, default: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        n = float(default)
    return round(max(0.0, min(1.0, n)), 2)


def _clamp099(value: Any, default: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        n = float(default)
    return round(max(0.0, min(0.99, n)), 2)


def _default_text_match_thresholds() -> dict[str, dict[str, float]]:
    return {
        k: {"match": _DEFAULT_MATCH_THR, "similar": _DEFAULT_SIMILAR_THR}
        for k in TEXT_MATCH_ALL
    }


def _normalize_one_threshold_pair(raw: Any) -> dict[str, float]:
    src = raw if isinstance(raw, dict) else {}
    match = _clamp01(src.get("match"), _DEFAULT_MATCH_THR)
    similar = _clamp01(src.get("similar"), _DEFAULT_SIMILAR_THR)
    if similar > match:
        similar = match
    return {"match": match, "similar": similar}


def _normalize_text_match_thresholds(value: Any) -> dict[str, dict[str, float]]:
    base = _default_text_match_thresholds()
    if not isinstance(value, dict):
        return base
    for key in TEXT_MATCH_ALL:
        if key in value:
            base[key] = _normalize_one_threshold_pair(value[key])
    return base


def score_band(
    score: float | None,
    *,
    match: float,
    similar: float,
) -> str:
    """Return ``match`` | ``similar`` | ``none`` for a text-match score."""
    if score is None:
        return "none"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "none"
    if s > float(match):
        return "match"
    if s < float(similar):
        return "none"
    return "similar"


def _normalize_embedding_device(value: Any) -> str:
    raw = str(value or "cpu").strip().lower()
    return "gpu" if raw == "gpu" else "cpu"


def _normalize_max_side(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 1024
    return n if n in NORMALIZE_MAX_SIDE_OPTIONS else 1024


def _normalize_text_match_methods(value: Any) -> list[str]:
    if not isinstance(value, list):
        return list(TEXT_MATCH_DEFAULT)
    allowed = set(TEXT_MATCH_ALL)
    out = [str(x) for x in value if str(x) in allowed]
    # preserve canonical order
    ordered = [k for k in TEXT_MATCH_ALL if k in set(out)]
    # local/server Cross Encoder — взаимоисключающие
    if "crenc" in ordered and "crenc_srv" in ordered:
        # оставляем тот, что ближе к концу исходного списка (последний выбор)
        last = None
        for x in out:
            if x in ("crenc", "crenc_srv"):
                last = x
        drop = "crenc_srv" if last == "crenc" else "crenc"
        ordered = [k for k in ordered if k != drop]
    return ordered


def _normalize_final_score_method(value: Any) -> str:
    raw = str(value or "fin1").strip().lower()
    return raw if raw in TEXT_MATCH_ALL else "fin1"


def _normalize_final_ocr(value: Any) -> str:
    raw = str(value or "auto").strip().lower().replace("-", "_")
    return raw if raw in FINAL_OCR_ALL else "auto"


def _normalize_hf_start_timeout(value: Any) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 15.0
    # 1..300 sec
    return max(1.0, min(300.0, n))


def _normalize(raw: dict[str, Any] | None) -> dict[str, Any]:
    base = _defaults()
    data = raw or {}
    prep = data.get("ocr_preprocess")
    if isinstance(prep, list):
        # Empty list is allowed (skip classical OCR preprocess/variants).
        base["ocr_preprocess"] = [
            p for p in PREPROCESS_ALL if p in {str(x).upper() for x in prep}
        ]
    eng = data.get("ocr_engines")
    if isinstance(eng, list):
        # Empty list is allowed (classical engines off; LLM OCR may still run).
        eng_set = {str(x).lower() for x in eng}
        base["ocr_engines"] = [
            e for e in ENGINE_ALL if e in eng_set and e not in ENGINE_DISABLED
        ]
    tw = data.get("text_weights")
    if isinstance(tw, dict):
        for key in WEIGHT_KEYS:
            if key in tw:
                try:
                    base["text_weights"][key] = max(0.0, float(tw[key]))
                except (TypeError, ValueError):
                    pass
    fw = data.get("final_weights")
    if isinstance(fw, dict):
        for key in FINAL_WEIGHT_KEYS:
            if key in fw:
                try:
                    base["final_weights"][key] = max(0.0, float(fw[key]))
                except (TypeError, ValueError):
                    pass
    if "use_dinov3" in data:
        base["use_dinov3"] = _as_bool(data["use_dinov3"], base["use_dinov3"])
    if "geometry_siglip2" in data:
        base["geometry_siglip2"] = _as_bool(
            data["geometry_siglip2"], base["geometry_siglip2"]
        )
    if "geometry_dinov3" in data:
        base["geometry_dinov3"] = _as_bool(
            data["geometry_dinov3"], base["geometry_dinov3"]
        )
    if "use_gemini_ocr" in data:
        base["use_gemini_ocr"] = _as_bool(
            data["use_gemini_ocr"], base["use_gemini_ocr"]
        )
    if "use_openai_ocr" in data:
        base["use_openai_ocr"] = _as_bool(
            data["use_openai_ocr"], base["use_openai_ocr"]
        )
    if "use_deepseek_ocr" in data:
        base["use_deepseek_ocr"] = _as_bool(
            data["use_deepseek_ocr"], base["use_deepseek_ocr"]
        )
    if "use_qwen_ocr" in data:
        base["use_qwen_ocr"] = _as_bool(
            data["use_qwen_ocr"], base["use_qwen_ocr"]
        )
    if "use_yandex_ocr" in data:
        base["use_yandex_ocr"] = _as_bool(
            data["use_yandex_ocr"], base["use_yandex_ocr"]
        )
    if "use_google_vision_ocr" in data:
        base["use_google_vision_ocr"] = _as_bool(
            data["use_google_vision_ocr"], base["use_google_vision_ocr"]
        )
    if "yolo_fallback_full_image" in data:
        base["yolo_fallback_full_image"] = _as_bool(
            data["yolo_fallback_full_image"], base["yolo_fallback_full_image"]
        )
    if "use_yolo" in data:
        base["use_yolo"] = _as_bool(data["use_yolo"], base["use_yolo"])
    if "yolo_variant" in data:
        base["yolo_variant"] = _normalize_yolo_variant(data["yolo_variant"])
    if "bottle_min_conf" in data:
        base["bottle_min_conf"] = _clamp01(
            data["bottle_min_conf"], base.get("bottle_min_conf", 0.60)
        )
    if "use_bottle_orient" in data:
        base["use_bottle_orient"] = _as_bool(
            data["use_bottle_orient"], base.get("use_bottle_orient", True)
        )
    if "bottle_orient_min_deg" in data:
        try:
            deg = float(data["bottle_orient_min_deg"])
        except (TypeError, ValueError):
            deg = float(base.get("bottle_orient_min_deg", 8.0))
        base["bottle_orient_min_deg"] = max(1.0, min(45.0, deg))
    if "label_detect_openai" in data:
        base["label_detect_openai"] = _as_bool(
            data["label_detect_openai"], base["label_detect_openai"]
        )
    if "label_detect_gemini" in data:
        base["label_detect_gemini"] = _as_bool(
            data["label_detect_gemini"], base["label_detect_gemini"]
        )
    if "embedding_device" in data:
        base["embedding_device"] = _normalize_embedding_device(
            data["embedding_device"]
        )
    if "embed_cpu_use_cache" in data:
        base["embed_cpu_use_cache"] = _as_bool(
            data["embed_cpu_use_cache"], base["embed_cpu_use_cache"]
        )
    if "normalize_max_side" in data:
        base["normalize_max_side"] = _normalize_max_side(
            data["normalize_max_side"]
        )
    if "hf_start_timeout_sec" in data:
        base["hf_start_timeout_sec"] = _normalize_hf_start_timeout(
            data["hf_start_timeout_sec"]
        )
    if "text_match_methods" in data:
        base["text_match_methods"] = _normalize_text_match_methods(
            data["text_match_methods"]
        )
    if "final_score_method" in data:
        base["final_score_method"] = _normalize_final_score_method(
            data["final_score_method"]
        )
    if "text_match_thresholds" in data:
        base["text_match_thresholds"] = _normalize_text_match_thresholds(
            data["text_match_thresholds"]
        )
    if "empty_ocr_cosine_threshold" in data:
        base["empty_ocr_cosine_threshold"] = _clamp099(
            data["empty_ocr_cosine_threshold"],
            base.get("empty_ocr_cosine_threshold", 0.86),
        )
    if "xgb_dead_max" in data:
        base["xgb_dead_max"] = _clamp099(
            data["xgb_dead_max"],
            base.get("xgb_dead_max", 0.15),
        )
    if "compute_hsv" in data:
        base["compute_hsv"] = _as_bool(
            data["compute_hsv"], base.get("compute_hsv", False)
        )
    if "use_hsv_filter" in data:
        base["use_hsv_filter"] = _as_bool(
            data["use_hsv_filter"], base.get("use_hsv_filter", False)
        )
    if "hsv_bhattacharyya_max" in data:
        base["hsv_bhattacharyya_max"] = _clamp01(
            data["hsv_bhattacharyya_max"],
            base.get("hsv_bhattacharyya_max", 0.50),
        )
    if "hsv_ignore_high_cosine" in data:
        base["hsv_ignore_high_cosine"] = _as_bool(
            data["hsv_ignore_high_cosine"],
            base.get("hsv_ignore_high_cosine", False),
        )
    if "hsv_ignore_cosine_min" in data:
        base["hsv_ignore_cosine_min"] = _clamp099(
            data["hsv_ignore_cosine_min"],
            base.get("hsv_ignore_cosine_min", 0.85),
        )
    if "use_hsv_hard_reject" in data:
        base["use_hsv_hard_reject"] = _as_bool(
            data["use_hsv_hard_reject"],
            base.get("use_hsv_hard_reject", False),
        )
    if "hsv_hard_reject_max" in data:
        base["hsv_hard_reject_max"] = _clamp01(
            data["hsv_hard_reject_max"],
            base.get("hsv_hard_reject_max", 0.90),
        )
    if "compute_color_delta" in data:
        base["compute_color_delta"] = _as_bool(
            data["compute_color_delta"],
            base.get("compute_color_delta", True),
        )
    if "reuse_previous_searches" in data:
        base["reuse_previous_searches"] = _as_bool(
            data["reuse_previous_searches"],
            base.get("reuse_previous_searches", False),
        )
    if "show_search_details" in data:
        base["show_search_details"] = _as_bool(
            data["show_search_details"],
            base.get("show_search_details", True),
        )
    # HSV gates require compute_hsv; keep defaults if master is off.
    if not bool(base.get("compute_hsv", False)):
        base["use_hsv_filter"] = False
        base["hsv_ignore_high_cosine"] = False
        base["use_hsv_hard_reject"] = False
    if "final_ocr" in data:
        base["final_ocr"] = _normalize_final_ocr(data["final_ocr"])
    if "exclusive_use_translit" in data:
        base["exclusive_use_translit"] = _as_bool(
            data["exclusive_use_translit"],
            base.get("exclusive_use_translit", True),
        )
    if "exclusive_match_spaced" in data:
        base["exclusive_match_spaced"] = _as_bool(
            data["exclusive_match_spaced"],
            base.get("exclusive_match_spaced", True),
        )
    # Mutual exclusions (score gates):
    # - HSV soft threshold ≤ hard reject (hard is stricter absolute cut)
    # - xgb_dead_max ≤ XGB match (dead must be below match band)
    thr = base.get("text_match_thresholds") or {}
    xgb_thr = thr.get("xgb") if isinstance(thr, dict) else None
    if isinstance(xgb_thr, dict):
        try:
            xgb_match = float(xgb_thr.get("match", 0.55))
        except (TypeError, ValueError):
            xgb_match = 0.55
        dead = _clamp099(base.get("xgb_dead_max"), 0.15)
        if dead > xgb_match:
            base["xgb_dead_max"] = _clamp099(xgb_match, 0.0)
    soft = _clamp01(base.get("hsv_bhattacharyya_max"), 0.50)
    hard = _clamp01(base.get("hsv_hard_reject_max"), 0.90)
    if bool(base.get("use_hsv_filter")) and bool(base.get("use_hsv_hard_reject")):
        if soft > hard:
            base["hsv_hard_reject_max"] = soft
    return base


def _read_file() -> dict[str, Any] | None:
    if not SETTINGS_PATH.is_file():
        return None
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_file(data: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _apply_runtime_config(data: dict[str, Any]) -> None:
    """Apply settings to in-memory app_config (no .env writes)."""
    app_config.OCR_PREPROCESS = list(data["ocr_preprocess"])
    app_config.OCR_ENGINES = list(data["ocr_engines"])
    tw = data["text_weights"]
    app_config.TEXT_WEIGHT_WINE = float(tw.get("wine", 0.35))
    app_config.TEXT_WEIGHT_PRODUCER = float(tw.get("producer", 0.28))
    app_config.TEXT_WEIGHT_COLOR = float(tw.get("color", 0.14))
    app_config.TEXT_WEIGHT_TYPE = float(tw.get("type", 0.08))
    app_config.TEXT_WEIGHT_GRAPE = float(tw.get("grape", 0.12))
    app_config.TEXT_WEIGHT_VINTAGE = float(tw.get("vintage", 0.12))
    app_config.TEXT_WEIGHT_REGION = float(tw.get("region", 0.08))
    app_config.TEXT_WEIGHT_OTHER = float(tw.get("other", 0.05))
    app_config.TEXT_WEIGHT_BRAND = float(tw.get("brand", 0.12))
    fw = data.get("final_weights") or {}
    app_config.FINAL_SCORE_W_TEXT = float(fw.get("text", 0.45))
    app_config.FINAL_SCORE_W_COS = float(fw.get("cosine", 0.55))
    app_config.USE_DINOV3 = bool(data["use_dinov3"])
    app_config.GEOMETRY_SIGLIP2 = bool(data["geometry_siglip2"])
    app_config.GEOMETRY_DINOV3 = bool(data["geometry_dinov3"])
    app_config.USE_GEMINI_OCR = bool(data["use_gemini_ocr"])
    app_config.USE_OPENAI_OCR = bool(data["use_openai_ocr"])
    app_config.USE_DEEPSEEK_OCR = bool(data["use_deepseek_ocr"])
    app_config.USE_QWEN_OCR = bool(data["use_qwen_ocr"])
    app_config.USE_YANDEX_OCR = bool(data["use_yandex_ocr"])
    app_config.USE_GOOGLE_VISION_OCR = bool(data["use_google_vision_ocr"])
    app_config.YOLO_FALLBACK_FULL_IMAGE = bool(data["yolo_fallback_full_image"])
    app_config.USE_YOLO = bool(data["use_yolo"])
    app_config.YOLO_VARIANT = _normalize_yolo_variant(data.get("yolo_variant"))
    app_config.LABEL_DETECT_OPENAI = bool(data["label_detect_openai"])
    app_config.LABEL_DETECT_GEMINI = bool(data["label_detect_gemini"])
    app_config.EMBEDDING_DEVICE = _normalize_embedding_device(
        data["embedding_device"]
    )
    app_config.EMBED_CPU_USE_CACHE = bool(data.get("embed_cpu_use_cache", False))
    app_config.NORMALIZE_MAX_SIDE = _normalize_max_side(
        data["normalize_max_side"]
    )
    app_config.HF_START_TIMEOUT_SEC = _normalize_hf_start_timeout(
        data["hf_start_timeout_sec"]
    )


def get_settings() -> dict[str, Any]:
    with _lock:
        data = _normalize(_read_file())
        # Синхронизируем app_config с pipeline_settings.json
        # (после reload worker иначе остаётся EMBEDDING_DEVICE из .env)
        _apply_runtime_config(data)
        return data


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        current = _normalize(_read_file())
        merged = deepcopy(current)
        if "ocr_preprocess" in patch:
            merged["ocr_preprocess"] = patch["ocr_preprocess"]
        if "ocr_engines" in patch:
            merged["ocr_engines"] = patch["ocr_engines"]
        if "text_weights" in patch and isinstance(patch["text_weights"], dict):
            merged["text_weights"].update(patch["text_weights"])
        if "final_weights" in patch and isinstance(patch["final_weights"], dict):
            merged.setdefault("final_weights", {})
            merged["final_weights"].update(patch["final_weights"])
        for key in (
            "use_dinov3",
            "geometry_siglip2",
            "geometry_dinov3",
            "use_gemini_ocr",
            "use_openai_ocr",
            "use_deepseek_ocr",
            "use_qwen_ocr",
            "use_yandex_ocr",
            "use_google_vision_ocr",
            "yolo_fallback_full_image",
            "use_yolo",
            "yolo_variant",
            "bottle_min_conf",
            "use_bottle_orient",
            "bottle_orient_min_deg",
            "label_detect_openai",
            "label_detect_gemini",
            "embedding_device",
            "embed_cpu_use_cache",
            "normalize_max_side",
            "hf_start_timeout_sec",
            "text_match_methods",
            "final_score_method",
            "empty_ocr_cosine_threshold",
            "xgb_dead_max",
            "compute_hsv",
            "use_hsv_filter",
            "hsv_bhattacharyya_max",
            "hsv_ignore_high_cosine",
            "hsv_ignore_cosine_min",
            "use_hsv_hard_reject",
            "hsv_hard_reject_max",
            "compute_color_delta",
            "reuse_previous_searches",
            "show_search_details",
            "final_ocr",
            "exclusive_use_translit",
            "exclusive_match_spaced",
        ):
            if key in patch:
                merged[key] = patch[key]
        if "text_match_thresholds" in patch and isinstance(
            patch["text_match_thresholds"], dict
        ):
            merged.setdefault("text_match_thresholds", {})
            for mk, mv in patch["text_match_thresholds"].items():
                if mk not in TEXT_MATCH_ALL:
                    continue
                cur = dict(merged["text_match_thresholds"].get(mk) or {})
                if isinstance(mv, dict):
                    cur.update(mv)
                merged["text_match_thresholds"][mk] = cur
        data = _normalize(merged)
        _write_file(data)
        _apply_runtime_config(data)
        return data


def get_ocr_preprocess() -> list[str]:
    """Active preprocess keys; may be empty (skip classical OCR variants)."""
    return list(get_settings()["ocr_preprocess"])


def get_ocr_engines() -> list[str]:
    """Active classical OCR engines; may be empty (LLM OCR only)."""
    return list(get_settings()["ocr_engines"])


def get_text_weights() -> dict[str, float]:
    return dict(get_settings()["text_weights"])


def get_final_weights() -> dict[str, float]:
    return dict(get_settings()["final_weights"])


def settings_public_view(data: dict[str, Any] | None = None) -> dict[str, Any]:
    s = data or get_settings()
    thr = _normalize_text_match_thresholds(s.get("text_match_thresholds"))
    return {
        "ocr_preprocess": s["ocr_preprocess"],
        "ocr_preprocess_options": [
            {**PREPROCESS_META[k], "enabled": k in set(s["ocr_preprocess"])}
            for k in PREPROCESS_ALL
        ],
        "ocr_engines": s["ocr_engines"],
        "ocr_engine_options": [
            {
                **ENGINE_META[k],
                "enabled": k in set(s["ocr_engines"]) and k not in ENGINE_DISABLED,
                "disabled": k in ENGINE_DISABLED,
            }
            for k in ENGINE_ALL
        ],
        "text_weights": s["text_weights"],
        "text_weight_options": [
            {
                **WEIGHT_META[k],
                "value": float(s["text_weights"][k]),
            }
            for k in WEIGHT_KEYS
        ],
        "final_weights": s["final_weights"],
        "final_weight_options": [
            {
                **FINAL_WEIGHT_META[k],
                "value": float(s["final_weights"][k]),
            }
            for k in FINAL_WEIGHT_KEYS
        ],
        "use_dinov3": bool(s["use_dinov3"]),
        "geometry_siglip2": bool(s["geometry_siglip2"]),
        "geometry_dinov3": bool(s["geometry_dinov3"]),
        "use_gemini_ocr": bool(s["use_gemini_ocr"]),
        "use_openai_ocr": bool(s["use_openai_ocr"]),
        "use_deepseek_ocr": bool(s["use_deepseek_ocr"]),
        "use_qwen_ocr": bool(s["use_qwen_ocr"]),
        "use_yandex_ocr": bool(s["use_yandex_ocr"]),
        "use_google_vision_ocr": bool(s["use_google_vision_ocr"]),
        "yolo_fallback_full_image": bool(s["yolo_fallback_full_image"]),
        "use_yolo": bool(s["use_yolo"]),
        "yolo_variant": _normalize_yolo_variant(s.get("yolo_variant")),
        "bottle_min_conf": _clamp01(s.get("bottle_min_conf"), 0.60),
        "use_bottle_orient": bool(s.get("use_bottle_orient", True)),
        "bottle_orient_min_deg": max(
            1.0, min(45.0, float(s.get("bottle_orient_min_deg", 8.0) or 8.0))
        ),
        "label_detect_openai": bool(s["label_detect_openai"]),
        "label_detect_gemini": bool(s["label_detect_gemini"]),
        "embedding_device": _normalize_embedding_device(s["embedding_device"]),
        "embed_cpu_use_cache": bool(s.get("embed_cpu_use_cache", False)),
        "normalize_max_side": _normalize_max_side(s["normalize_max_side"]),
        "normalize_max_side_options": list(NORMALIZE_MAX_SIDE_OPTIONS),
        "hf_start_timeout_sec": _normalize_hf_start_timeout(
            s["hf_start_timeout_sec"]
        ),
        "text_match_methods": list(s.get("text_match_methods") or TEXT_MATCH_ALL),
        "text_match_thresholds": thr,
        "text_match_options": [
            {
                **TEXT_MATCH_META[k],
                "enabled": k in set(s.get("text_match_methods") or []),
                "match": float(thr[k]["match"]),
                "similar": float(thr[k]["similar"]),
            }
            for k in TEXT_MATCH_ALL
        ],
        "final_score_method": _normalize_final_score_method(
            s.get("final_score_method")
        ),
        "final_score_method_options": [
            {"id": k, "label": TEXT_MATCH_META[k]["label"]} for k in TEXT_MATCH_ALL
        ],
        "empty_ocr_cosine_threshold": _clamp099(
            s.get("empty_ocr_cosine_threshold"), 0.86
        ),
        "xgb_dead_max": _clamp099(s.get("xgb_dead_max"), 0.15),
        "compute_hsv": bool(s.get("compute_hsv", False)),
        "use_hsv_filter": bool(s.get("use_hsv_filter", False)),
        "hsv_bhattacharyya_max": _clamp01(
            s.get("hsv_bhattacharyya_max"), 0.50
        ),
        "hsv_ignore_high_cosine": bool(s.get("hsv_ignore_high_cosine", False)),
        "hsv_ignore_cosine_min": _clamp099(
            s.get("hsv_ignore_cosine_min"), 0.85
        ),
        "use_hsv_hard_reject": bool(s.get("use_hsv_hard_reject", False)),
        "hsv_hard_reject_max": _clamp01(s.get("hsv_hard_reject_max"), 0.90),
        "compute_color_delta": bool(s.get("compute_color_delta", True)),
        "reuse_previous_searches": bool(s.get("reuse_previous_searches", False)),
        "show_search_details": bool(s.get("show_search_details", True)),
        "final_ocr": _normalize_final_ocr(s.get("final_ocr")),
        "final_ocr_options": [
            {"id": k, "label": FINAL_OCR_META[k]["label"]} for k in FINAL_OCR_ALL
        ],
        "exclusive_use_translit": bool(s.get("exclusive_use_translit", True)),
        "exclusive_match_spaced": bool(s.get("exclusive_match_spaced", True)),
    }
