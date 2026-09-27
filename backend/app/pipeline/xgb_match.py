"""XGBoost OCR↔label matcher — отдельный шаг findwine после top-20 embedding.

Модель: XGB_MODEL_DIR (default backend/models/xgboost_text_matcher_synthetic_40_60/).
Признаки: app.pipeline.xgb_text_features (копия text_features.py).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.db import config as app_config
from app.pipeline.xgb_text_features import WineRef, group_features

_log = logging.getLogger("xgb_match")


def _model_dir() -> Path:
    """Каталог модели из XGB_MODEL_DIR (.env / app_config)."""
    raw = getattr(app_config, "XGB_MODEL_DIR", None)
    if raw is None or str(raw).strip() == "":
        return (
            Path(__file__).resolve().parents[2]
            / "models"
            / "xgboost_text_matcher_synthetic_40_60"
        )
    return Path(raw).resolve()

_lock = threading.Lock()
_model = None
_feature_names: list[str] | None = None
_threshold: float = 0.29
_loaded_dir: str | None = None

# Приоритет OCR как при обучении (prepare_dataset OCR_ENGINES_DEFAULT)
_OCR_ENGINE_PRIORITY = (
    "google_vision",
    "gemini",
    "openai",
    "yandex",
    "deepseek",
    "qwen",
)

# Движки, которые можно выбрать как Final OCR в настройках
FINAL_OCR_ENGINES = _OCR_ENGINE_PRIORITY

_OCR_STEP_KEYS = {
    "google_vision": "ocr_google_vision",
    "gemini": "ocr_gemini",
    "openai": "ocr_openai",
    "yandex": "ocr_yandex",
    "deepseek": "ocr_deepseek",
    "qwen": "ocr_qwen",
}


def _normalize_final_ocr(value: Any) -> str:
    raw = str(value or "auto").strip().lower().replace("-", "_")
    if raw in {"auto", ""}:
        return "auto"
    if raw in FINAL_OCR_ENGINES:
        return raw
    return "auto"


def _text_from_ocr_variant(var: dict[str, Any]) -> str:
    lines_out: list[str] = []
    raw_lines = var.get("lines")
    if isinstance(raw_lines, list):
        for ln in raw_lines:
            if isinstance(ln, dict):
                t = str(ln.get("text") or "").strip()
            else:
                t = str(ln or "").strip()
            if t:
                lines_out.append(t)
    if not lines_out:
        txt = str(var.get("text") or "").strip()
        if txt:
            lines_out = [x.strip() for x in txt.splitlines() if x.strip()]
    return "\n".join(lines_out)


def list_ocr_texts_by_engine(status: dict[str, Any] | None) -> dict[str, str]:
    """{engine: text} для cloud/LLM OCR с непустым текстом."""
    out: dict[str, str] = {}
    if not isinstance(status, dict):
        return out
    steps = status.get("steps") or {}
    ocr = steps.get("ocr") or {}
    variants = ocr.get("variants") if isinstance(ocr, dict) else None
    if isinstance(variants, dict):
        for eng in FINAL_OCR_ENGINES:
            var = variants.get(eng)
            if not isinstance(var, dict) or not var.get("ok"):
                continue
            txt = _text_from_ocr_variant(var)
            if txt.strip():
                out[eng] = txt
    for eng, step_key in _OCR_STEP_KEYS.items():
        if eng in out:
            continue
        step = steps.get(step_key) or {}
        if not isinstance(step, dict):
            continue
        txt = str(step.get("text") or step.get("text_aggregated") or "").strip()
        if txt:
            out[eng] = txt
    return out


def resolve_final_ocr(
    status: dict[str, Any] | None,
    prefer: str | None = None,
) -> tuple[str | None, str]:
    """Вернуть (engine|None, text). prefer=auto → приоритет обучения."""
    texts = list_ocr_texts_by_engine(status)
    pref = _normalize_final_ocr(prefer)
    if pref != "auto" and pref in texts:
        return pref, texts[pref]
    for eng in _OCR_ENGINE_PRIORITY:
        if eng in texts:
            return eng, texts[eng]
    if isinstance(status, dict):
        steps = status.get("steps") or {}
        ocr = steps.get("ocr") or {}
        if isinstance(ocr, dict):
            for k in ("text", "text_aggregated", "text_norm"):
                t = str(ocr.get(k) or "").strip()
                if t:
                    return None, t
    return None, ""


def collect_query_ocr_text(
    status: dict[str, Any] | None,
    *,
    prefer: str | None = None,
) -> str:
    """Сырой OCR запроса: prefer=engine|auto (приоритет как при обучении)."""
    _eng, text = resolve_final_ocr(status, prefer=prefer)
    return text


def model_dir() -> Path:
    return _model_dir()


def _load_threshold(dir_path: Path) -> float:
    metrics_path = dir_path / "metrics.json"
    if not metrics_path.is_file():
        return 0.29
    try:
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        t = raw.get("threshold_best_f1_validation")
        if t is None:
            t = (raw.get("validation") or {}).get("threshold")
        if t is not None:
            return float(t)
    except Exception as exc:  # noqa: BLE001
        _log.warning("xgb threshold load failed: %s", exc)
    return 0.29


def _ensure_model() -> tuple[Any, list[str], float]:
    global _model, _feature_names, _threshold, _loaded_dir
    with _lock:
        model_dir = _model_dir()
        d = str(model_dir)
        if _model is not None and _feature_names is not None and _loaded_dir == d:
            return _model, _feature_names, _threshold
        model_path = model_dir / "model.json"
        names_path = model_dir / "feature_names.json"
        if not model_path.is_file():
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")
        if not names_path.is_file():
            raise FileNotFoundError(f"feature_names.json not found: {names_path}")
        from xgboost import XGBClassifier

        names = json.loads(names_path.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not names:
            raise ValueError("feature_names.json empty")
        clf = XGBClassifier()
        clf.load_model(str(model_path))
        thr = _load_threshold(model_dir)
        _model = clf
        _feature_names = [str(x) for x in names]
        _threshold = thr
        _loaded_dir = d
        _log.info(
            "XGBoost loaded dir=%s features=%s threshold=%.3f",
            model_dir,
            len(_feature_names),
            _threshold,
        )
        return _model, _feature_names, _threshold


def get_xgb_threshold() -> float:
    """Порог suitable из metrics.json (или default)."""
    try:
        _, _, thr = _ensure_model()
        return float(thr)
    except Exception:  # noqa: BLE001
        return 0.29


def _label_tokens(text: str | None) -> set[str]:
    """Токены этикетки для hard-reject (как в XGB features)."""
    from app.pipeline.xgb_text_features import normalize_text, tokens

    return set(tokens(normalize_text(text)))


def catalog_label_text(wine: dict[str, Any]) -> str:
    """Текст этикетки каталога для сравнения (поле label)."""
    return str(wine.get("label") or "").strip()


def label_text_hard_reject_reason(
    ocr_text: str | None,
    wine: dict[str, Any],
) -> str | None:
    """Правила hard-reject по тексту этикеток.

    1. оба текста непусты и нет ни одного общего слова → no_shared_words
    2. OCR есть, у кандидата label пустой → query_text_catalog_empty
    3. OCR пустой, у кандидата label есть → query_empty_catalog_text

    Оба пустые — не reject (сравнение «только картинка»).
    Сравнение токенов всегда в нижнем регистре (normalize_text → lower).
    """
    from app.pipeline.xgb_text_features import normalize_text

    q_norm = normalize_text(str(ocr_text or ""))
    c_norm = normalize_text(catalog_label_text(wine))
    q_tok = _label_tokens(q_norm)
    c_tok = _label_tokens(c_norm)
    q_has = bool(q_tok)
    c_has = bool(c_tok)
    if q_has and not c_has:
        return "query_text_catalog_empty"
    if (not q_has) and c_has:
        return "query_empty_catalog_text"
    if q_has and c_has and q_tok.isdisjoint(c_tok):
        return "no_shared_words"
    return None


_LABEL_TEXT_REJECT_TITLES = {
    "no_shared_words": "нет общих слов OCR↔этикетка",
    "query_text_catalog_empty": "OCR есть, у кандидата нет текста этикетки",
    "query_empty_catalog_text": "OCR пустой, у кандидата есть текст этикетки",
}


def evaluate_label_text_hard_reject(
    ocr_text: str | None,
    wines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Hard-reject кандидатов по пересечению/наличию текста этикеток."""
    t0 = time.perf_counter()
    rejected: list[dict[str, Any]] = []
    rejected_ids: list[int] = []
    reasons: dict[str, str] = {}
    for w in wines:
        if w.get("id") is None:
            continue
        reason = label_text_hard_reject_reason(ocr_text, w)
        if not reason:
            continue
        wid = int(w["id"])
        rejected_ids.append(wid)
        reasons[str(wid)] = reason
        rejected.append(
            {
                "wine_id": wid,
                "name": w.get("name"),
                "reason": reason,
                "reason_title": _LABEL_TEXT_REJECT_TITLES.get(reason, reason),
            }
        )
    return {
        "ok": True,
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "rejected_ids": rejected_ids,
        "reject_count": len(rejected_ids),
        "rejected": rejected[:40],
        "reasons": reasons,
        "rules": [
            "no shared tokens (len≥2) between query OCR and catalog label → reject",
            "query OCR non-empty, catalog label empty → reject",
            "query OCR empty, catalog label non-empty → reject",
        ],
    }


def wine_row_to_ref(w: dict[str, Any]) -> WineRef:
    def _year(value: Any) -> int | None:
        try:
            y = int(value)
        except (TypeError, ValueError):
            return None
        return y if 1950 <= y <= 2100 else None

    lo = w.get("label_ocr") if isinstance(w.get("label_ocr"), dict) else {}
    return WineRef(
        wine_id=int(w["id"]),
        label=w.get("label"),
        name=w.get("name"),
        winery=w.get("winery"),
        category=w.get("category"),
        color=w.get("color"),
        grape=w.get("grape_variety") or w.get("grape"),
        region=w.get("region"),
        label_color=(
            w.get("label_color")
            or (str(lo["color"]) if lo.get("color") not in (None, "") else None)
        ),
        label_type=(
            w.get("label_type")
            or (str(lo["type"]) if lo.get("type") not in (None, "") else None)
        ),
        vintage_year=_year(
            w.get("vintage_year")
            if w.get("vintage_year") is not None
            else lo.get("vintage_year")
        ),
        foundation_year=_year(
            w.get("foundation_year")
            if w.get("foundation_year") is not None
            else lo.get("foundation_year")
        ),
    )


def enrich_wine_dict_for_xgb(w: dict[str, Any]) -> dict[str, Any]:
    """Гарантировать поля для фич v1.3 (label_ocr color/type/vintage)."""
    out = dict(w)
    lo = out.get("label_ocr") if isinstance(out.get("label_ocr"), dict) else {}
    if out.get("label_color") in (None, "") and lo.get("color") not in (None, ""):
        out["label_color"] = str(lo["color"])
    if out.get("label_type") in (None, "") and lo.get("type") not in (None, ""):
        out["label_type"] = str(lo["type"])
    if out.get("vintage_year") is None and lo.get("vintage_year") is not None:
        out["vintage_year"] = lo.get("vintage_year")
    if out.get("foundation_year") is None and lo.get("foundation_year") is not None:
        out["foundation_year"] = lo.get("foundation_year")
    if lo and not isinstance(out.get("label_ocr"), dict):
        out["label_ocr"] = lo
    return out


def score_candidates(
    ocr_text: str,
    wines: list[dict[str, Any]],
    cosine_by_id: dict[int, float],
) -> list[dict[str, Any]]:
    """Вернуть [{id, xgb_score, ...}] в том же порядке, что wines.

    Считает модель по всем кандидатам (в т.ч. с hard reject) — чтобы в UI
    было видно реальный XGB; обнуление fin делает ``apply_xgb_fin``.
    """
    if not wines:
        return []
    model, feature_names, _thr = _ensure_model()
    enriched = [enrich_wine_dict_for_xgb(w) for w in wines]
    refs = [wine_row_to_ref(w) for w in enriched]
    # cosines arg retained for API compat; absolute text features ignore it
    cosines = [cosine_by_id.get(int(w["id"])) for w in enriched]
    rows = group_features(ocr_text, refs, cosines)
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            v = row.get(name)
            try:
                fv = float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                fv = 0.0
            if fv != fv:  # NaN
                fv = 0.0
            X[i, j] = fv
    proba = model.predict_proba(X)[:, 1]
    out: list[dict[str, Any]] = []
    for w, p in zip(enriched, proba):
        out.append(
            {
                "id": int(w["id"]),
                "xgb_score": round(float(p), 6),
                "cosine": cosine_by_id.get(int(w["id"])),
            }
        )
    return out


def score_candidates_skipping_hard_reject(
    ocr_text: str,
    wines: list[dict[str, Any]],
    cosine_by_id: dict[int, float],
    hard_reject_ids: set[int] | frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    """Считать XGB по всем кандидатам; hard_reject только помечает флаг.

    Историческое имя: раньше skipped → score=0 без модели. Теперь модель
    всегда вызывается, чтобы видеть реальный xgb_score при reject.
    """
    rejected = {int(x) for x in (hard_reject_ids or set())}
    scored = score_candidates(ocr_text, wines, cosine_by_id)
    out: list[dict[str, Any]] = []
    for row in scored:
        wid = int(row["id"])
        item = dict(row)
        if wid in rejected:
            item["hard_rejected"] = True
        out.append(item)
    return out


def start_xgb_warmup_background() -> None:
    """Подгрузить модель до первого поиска, если XGBoost включён в настройках."""
    def _job() -> None:
        try:
            from app.pipeline.runtime_settings import get_settings

            methods = set(get_settings().get("text_match_methods") or [])
            if "xgb" not in methods:
                return
            _ensure_model()
        except Exception:  # noqa: BLE001
            _log.exception("xgb warmup failed")

    threading.Thread(target=_job, name="xgb-warmup", daemon=True).start()


def apply_xgb_fin(
    scored: list[dict[str, Any]],
    *,
    match_threshold: float,
    similar_threshold: float,
    exclusive_reject_ids: set[int] | frozenset[int] | None = None,
    label_text_reject_reasons: dict[int, str] | None = None,
    final_weights: dict[str, float] | None = None,
    cosine_by_id: dict[int, float] | None = None,
    # legacy alias: treated as match_threshold if match_threshold not meaningful
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """XGB = TextScore; XGB_fin как fin1/fin2.

    - TextScore (xgb_score) ≥ match → fin = w_ocr·XGB + w_emb·Cosine (suitable)
    - similar ≤ TextScore < match → fin = та же смесь (не победитель)
    - TextScore < similar → fin = 0
    - hard reject (exclusive / label-text / HSV): xgb_score остаётся реальным;
      xgb_fin = 0, xgb_fin_pre_reject = fin без reject (для UI серым)
    """
    from app.pipeline.ocr_match import compute_final_score, final_score_weights
    from app.pipeline.runtime_settings import score_band

    rejected = {int(x) for x in (exclusive_reject_ids or set())}
    text_rej = {
        int(k): str(v)
        for k, v in (label_text_reject_reasons or {}).items()
    }
    fw = final_weights or final_score_weights()
    cos_map = {int(k): float(v) for k, v in (cosine_by_id or {}).items()}
    match_t = float(
        match_threshold if match_threshold is not None else (threshold or 0.6)
    )
    similar_t = float(similar_threshold)
    out: list[dict[str, Any]] = []
    for row in scored:
        wid = int(row["id"])
        score = float(row.get("xgb_score") or 0.0)
        cos = row.get("cosine")
        if cos is None:
            cos = cos_map.get(wid)
        else:
            try:
                cos = float(cos)
            except (TypeError, ValueError):
                cos = cos_map.get(wid)
        text_reason = text_rej.get(wid)
        excl = wid in rejected or bool(text_reason)
        band_raw = score_band(score, match=match_t, similar=similar_t)
        if band_raw == "none":
            fin_raw = 0.0
        else:
            mixed = compute_final_score(
                text_score_01=score, cosine=cos, weights=fw
            )
            fin_raw = float(mixed["final_score"])
        if text_reason:
            fin = 0.0
            reason = f"label_text:{text_reason}"
            suitable = False
            band = "none"
        elif excl:
            fin = 0.0
            reason = "exclusive_lexicon"
            suitable = False
            band = "none"
        elif band_raw == "none":
            fin = 0.0
            reason = "below_similar"
            suitable = False
            band = "none"
        else:
            fin = fin_raw
            suitable = band_raw == "match"
            reason = "ok" if suitable else "similar_band"
            band = band_raw
        out.append(
            {
                **row,
                "xgb_score": round(float(score), 6),
                "cosine": None if cos is None else round(float(cos), 6),
                "band": band,
                "suitable": suitable,
                "exclusive_rejected": excl,
                "label_text_hard_reject": text_reason,
                "xgb_fin": round(fin, 6),
                "xgb_fin_pre_reject": round(fin_raw, 6),
                "xgb_fin_reason": reason,
            }
        )
    return out

def evaluate_xgb_match(
    *,
    ocr_text: str,
    wines: list[dict[str, Any]],
    cosine_by_id: dict[int, float],
    exclusive_reject_ids: set[int] | frozenset[int] | None = None,
) -> dict[str, Any]:
    """Полный шаг status.steps.xgb_match."""
    from app.pipeline.ocr_match import final_score_weights
    from app.pipeline.runtime_settings import (
        _normalize_text_match_thresholds,
        get_settings,
    )

    t0 = time.perf_counter()
    thr_pair = _normalize_text_match_thresholds(
        (get_settings() or {}).get("text_match_thresholds")
    ).get("xgb") or {"match": 0.6, "similar": 0.2}
    match_t = float(thr_pair["match"])
    similar_t = float(thr_pair["similar"])
    fw = final_score_weights()
    explain = {
        "model_dir": str(_model_dir()),
        "ocr_engines_priority": list(_OCR_ENGINE_PRIORITY),
        "text_score": "xgb_score = P(match) XGBoost (TextScore)",
        "band": (
            f"match if TextScore > {match_t}; similar if ≥ {similar_t}; "
            "else none"
        ),
        "xgb_fin": (
            "match|similar → w_ocr×XGB + w_emb×Cosine; below similar or "
            "exclusive → 0; winner only from match band"
        ),
        "final_weights": fw,
        "thresholds": {"match": match_t, "similar": similar_t},
    }
    text = str(ocr_text or "").strip()
    if not text:
        return {
            "ok": False,
            "error": "empty_ocr_text",
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "scores": [],
            "by_id": {},
            "explain": explain,
        }
    if not wines:
        return {
            "ok": False,
            "error": "no_candidates",
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "scores": [],
            "by_id": {},
            "explain": explain,
        }
    try:
        _, feature_names, _model_thr = _ensure_model()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "scores": [],
            "by_id": {},
            "explain": explain,
        }
    try:
        text_hr = evaluate_label_text_hard_reject(text, wines)
        text_reasons = {
            int(k): str(v)
            for k, v in (text_hr.get("reasons") or {}).items()
        }
        excl = set(int(x) for x in (exclusive_reject_ids or set()))
        excl |= set(text_reasons.keys())
        scored = score_candidates_skipping_hard_reject(
            text, wines, cosine_by_id, excl
        )
        ranked = apply_xgb_fin(
            scored,
            match_threshold=match_t,
            similar_threshold=similar_t,
            exclusive_reject_ids=excl,
            label_text_reject_reasons=text_reasons,
            final_weights=fw,
            cosine_by_id=cosine_by_id,
        )
    except Exception as exc:  # noqa: BLE001
        _log.exception("xgb_match failed")
        return {
            "ok": False,
            "error": str(exc),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "scores": [],
            "by_id": {},
            "threshold": match_t,
            "thresholds": {"match": match_t, "similar": similar_t},
            "explain": explain,
        }
    ranked_sorted = sorted(
        ranked,
        key=lambda r: (
            -float(r.get("xgb_fin") or 0),
            -float(r.get("xgb_score") or 0),
        ),
    )
    best_fin = next(
        (r for r in ranked_sorted if r.get("suitable") and float(r.get("xgb_fin") or 0) > 0),
        None,
    )
    by_id = {str(int(r["id"])): r for r in ranked}
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "ok": True,
        "ms": ms,
        "threshold": match_t,
        "thresholds": {"match": match_t, "similar": similar_t},
        "feature_count": len(feature_names),
        "n_candidates": len(ranked),
        "n_suitable": sum(1 for r in ranked if r.get("suitable")),
        "ocr_chars": len(text),
        "scores": ranked_sorted,
        "by_id": by_id,
        "best_xgb_fin": best_fin,
        "explain": explain,
    }


def attach_xgb_to_candidate_items(
    candidates: dict[str, Any] | None,
    xgb_step: dict[str, Any] | None,
) -> None:
    """Мутирует candidates.*.items: xgb_score / xgb_fin."""
    if not isinstance(candidates, dict) or not isinstance(xgb_step, dict):
        return
    by_id = xgb_step.get("by_id") or {}
    if not isinstance(by_id, dict):
        return
    for branch in ("siglip2", "dinov3"):
        block = candidates.get(branch) or {}
        items = block.get("items") if isinstance(block, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict) or it.get("id") is None:
                continue
            row = by_id.get(str(int(it["id"])))
            if not isinstance(row, dict):
                continue
            it["xgb_score"] = row.get("xgb_score")
            it["xgb_fin"] = row.get("xgb_fin")
            it["xgb_fin_pre_reject"] = row.get("xgb_fin_pre_reject")
            it["xgb_suitable"] = row.get("suitable")
            it["xgb_fin_reason"] = row.get("xgb_fin_reason")
            it["exclusive_rejected"] = row.get("exclusive_rejected")
            it["label_text_hard_reject"] = row.get("label_text_hard_reject")
