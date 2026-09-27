"""Cross-Encoder OCR↔label matcher — шаг findwine рядом с XGBoost.

Два бэкенда (взаимоисключающие в settings):
  local  — backend/models/cross_encoder/
  server — CRENC_ENDPOINT …/v1/match (тот же X-API-Key, что у SigLIP2)

text1/text2 — как при обучении (ocr_cross_encoder_text / WineRef.cross_encoder_text).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import requests

from app.db import config as app_config
from app.pipeline.xgb_match import wine_row_to_ref
from app.pipeline.xgb_text_features import ocr_cross_encoder_text

_log = logging.getLogger("crenc_match")

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "cross_encoder"

_lock = threading.Lock()
_model = None
_threshold: float = 0.26
_loaded_dir: str | None = None


def _load_threshold(dir_path: Path) -> float:
    metrics_path = dir_path / "metrics.json"
    if not metrics_path.is_file():
        return 0.26
    try:
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        t = raw.get("threshold_best_f1_validation")
        if t is None:
            t = (raw.get("validation") or {}).get("threshold")
        if t is not None:
            return float(t)
    except Exception as exc:  # noqa: BLE001
        _log.warning("crenc threshold load failed: %s", exc)
    return 0.26


def _ensure_model() -> tuple[Any, float]:
    global _model, _threshold, _loaded_dir
    with _lock:
        d = str(_MODEL_DIR)
        if _model is not None and _loaded_dir == d:
            return _model, _threshold
        if not (_MODEL_DIR / "config.json").is_file():
            raise FileNotFoundError(f"CrossEncoder model not found: {_MODEL_DIR}")
        from sentence_transformers import CrossEncoder

        # Как при обучении (train_config max_length=160), не дефолт базы 512.
        ce = CrossEncoder(str(_MODEL_DIR), max_length=160)
        thr = _load_threshold(_MODEL_DIR)
        _model = ce
        _threshold = thr
        _loaded_dir = d
        _log.info("CrossEncoder loaded dir=%s threshold=%.3f", _MODEL_DIR, _threshold)
        return _model, _threshold


def get_crenc_threshold() -> float:
    """Порог из metrics.json локальной модели, без загрузки весов."""
    return _load_threshold(_MODEL_DIR)


def _match_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/v1/match"):
        return base
    return f"{base}/v1/match"


def score_candidates_server(
    ocr_text: str,
    wines: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Remote CrossEncoder: POST {CRENC_ENDPOINT}/v1/match. Returns (rows, threshold)."""
    if not wines:
        return [], get_crenc_threshold()
    endpoint = (
        getattr(app_config, "CRENC_ENDPOINT", "")
        or "https://vino-svoe.online/api_cross_encoder_matcher"
    ).strip()
    url = _match_url(endpoint)
    api_key = (getattr(app_config, "EMBED_API_KEY", "") or "").strip()
    timeout = float(
        getattr(app_config, "CRENC_TIMEOUT_SEC", None)
        or getattr(app_config, "EMBED_TIMEOUT_SEC", 120)
        or 120
    )
    text1 = ocr_cross_encoder_text(ocr_text)
    refs = [wine_row_to_ref(w) for w in wines]
    pairs = [
        {
            "id": int(w["id"]),
            "text1": text1,
            "text2": ref.cross_encoder_text(),
        }
        for w, ref in zip(wines, refs)
    ]
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        resp = requests.post(
            url,
            headers=headers,
            json={"pairs": pairs, "batch_size": 32},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"CrossEncoder server timeout {timeout:g}s: {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"CrossEncoder server request failed: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"CrossEncoder server HTTP {resp.status_code}: {resp.text[:500]}"
        )
    data = resp.json()
    thr = float(
        data.get("threshold")
        if data.get("threshold") is not None
        else get_crenc_threshold()
    )
    items = data.get("items")
    scores = data.get("scores")
    by_id: dict[int, float] = {}
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict) or it.get("id") is None:
                continue
            try:
                by_id[int(it["id"])] = float(it.get("score") or 0.0)
            except (TypeError, ValueError):
                continue
    if not by_id and isinstance(scores, list) and len(scores) == len(wines):
        for w, sc in zip(wines, scores):
            by_id[int(w["id"])] = float(sc)
    out: list[dict[str, Any]] = []
    for w in wines:
        wid = int(w["id"])
        out.append(
            {
                "id": wid,
                "crenc_score": round(float(by_id.get(wid, 0.0)), 6),
            }
        )
    return out, thr


def score_candidates(
    ocr_text: str,
    wines: list[dict[str, Any]],
    *,
    backend: str = "local",
) -> list[dict[str, Any]]:
    """[{id, crenc_score, ...}] в порядке wines.

    backend: local | server. Для server предпочтительнее score_candidates_server
    (возвращает threshold с endpoint).
    """
    if str(backend).strip().lower() == "server":
        rows, _thr = score_candidates_server(ocr_text, wines)
        return rows
    if not wines:
        return []
    model, _thr = _ensure_model()
    text1 = ocr_cross_encoder_text(ocr_text)
    refs = [wine_row_to_ref(w) for w in wines]
    pairs = [[text1, ref.cross_encoder_text()] for ref in refs]
    # Один батч на top-20; max_length уже 160.
    probs = model.predict(pairs, batch_size=32, show_progress_bar=False)
    out: list[dict[str, Any]] = []
    for w, p in zip(wines, probs):
        out.append(
            {
                "id": int(w["id"]),
                "crenc_score": round(float(p), 6),
            }
        )
    return out


def start_crenc_warmup_background() -> None:
    """Подгрузить локальный CrossEncoder до первого поиска, если метод включён."""
    def _job() -> None:
        try:
            from app.pipeline.runtime_settings import get_settings

            methods = set(get_settings().get("text_match_methods") or [])
            if "crenc" not in methods:
                return
            _ensure_model()
        except Exception:  # noqa: BLE001
            _log.exception("crenc warmup failed")

    threading.Thread(target=_job, name="crenc-warmup", daemon=True).start()


def apply_crenc_fin(
    scored: list[dict[str, Any]],
    *,
    match_threshold: float,
    similar_threshold: float,
    exclusive_reject_ids: set[int] | frozenset[int] | None = None,
    final_weights: dict[str, float] | None = None,
    cosine_by_id: dict[int, float] | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """CrEnc = TextScore; CrEnc_fin как fin1/fin2.

    - TextScore (crenc_score) ≥ match → fin = w_ocr·CrEnc + w_emb·Cosine (suitable)
    - similar ≤ TextScore < match → fin = та же смесь (не победитель)
    - TextScore < similar или exclusive → fin = 0
    """
    from app.pipeline.ocr_match import compute_final_score, final_score_weights
    from app.pipeline.runtime_settings import score_band

    rejected = {int(x) for x in (exclusive_reject_ids or set())}
    fw = final_weights or final_score_weights()
    cos_map = {int(k): float(v) for k, v in (cosine_by_id or {}).items()}
    match_t = float(
        match_threshold if match_threshold is not None else (threshold or 0.55)
    )
    similar_t = float(similar_threshold)
    out: list[dict[str, Any]] = []
    for row in scored:
        wid = int(row["id"])
        score = float(row.get("crenc_score") or 0.0)
        cos = row.get("cosine")
        if cos is None:
            cos = cos_map.get(wid)
        else:
            try:
                cos = float(cos)
            except (TypeError, ValueError):
                cos = cos_map.get(wid)
        excl = wid in rejected
        band = score_band(score, match=match_t, similar=similar_t)
        if excl:
            fin = 0.0
            reason = "exclusive_lexicon"
            suitable = False
        elif band == "none":
            fin = 0.0
            reason = "below_similar"
            suitable = False
        else:
            mixed = compute_final_score(
                text_score_01=score, cosine=cos, weights=fw
            )
            fin = float(mixed["final_score"])
            suitable = band == "match"
            reason = "ok" if suitable else "similar_band"
        out.append(
            {
                **row,
                "cosine": None if cos is None else round(float(cos), 6),
                "band": band if not excl else "none",
                "suitable": suitable,
                "exclusive_rejected": excl,
                "crenc_fin": round(fin, 6),
                "crenc_fin_reason": reason,
            }
        )
    return out

def attach_crenc_to_candidate_items(
    candidates: dict[str, Any] | None,
    crenc_step: dict[str, Any] | None,
) -> None:
    if not isinstance(candidates, dict) or not isinstance(crenc_step, dict):
        return
    by_id = crenc_step.get("by_id") or {}
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
            it["crenc_score"] = row.get("crenc_score")
            it["crenc_fin"] = row.get("crenc_fin")
            it["crenc_suitable"] = row.get("suitable")
            it["crenc_fin_reason"] = row.get("crenc_fin_reason")
