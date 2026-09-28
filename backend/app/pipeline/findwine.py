"""Findwine pipeline: save >> YOLO|LLM-detect >> parallel(embed∥OCR) >> match."""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.config import EMBED_DIM, SEARCH_PHOTO_DIR, WORK_JPEG_MAX_BYTES
from app.db import config as app_config
from app.db.models import SearchPhoto, Wine, WineSitemap
from app.media import label_url, photo_url
from app.pipeline.analogs import (
    TYPE_LABELS,
    find_analogs,
    find_analogs_from_wine,
)
from app.pipeline.analogs import _type_canon as _analog_type_canon
from app.pipeline.candidates import search_top_wines_by_cosine
from app.pipeline.embed_client import (
    embed_dinov3,
    embed_siglip2,
    embed_siglip2_remote_cpu,
    get_hf_session,
    warm_hf_connection,
)
from app.pipeline.siglip_local import embed_siglip2_local, local_enabled as siglip_local_enabled
from app.pipeline.geometry import attach_geometry_to_items, verify_candidates_geometry
from app.pipeline.hsv_match import score_candidates_hsv
from app.pipeline.color_delta import score_candidates_color_delta
from app.pipeline.hf_endpoint_control import (
    is_hf_endpoint_network_error,
    is_hf_endpoint_paused_error,
    start_hf_endpoint,
)
from app.pipeline.image_io import (
    NORMALIZE_JPEG_QUALITY,
    align_filename_extension,
    normalize_image_bytes,
    open_image_rgb,
    save_rgb_jpeg,
    save_short_side_jpeg,
    save_work_jpeg_budget,
)
from app.pipeline.ocr import run_ocr_variants
from app.pipeline.ocr_gemini import run_gemini_label_detect_ocr, run_gemini_ocr
from app.pipeline.ocr_openai import run_openai_label_detect_ocr, run_openai_ocr
from app.pipeline.ocr_deepseek import run_deepseek_ocr
from app.pipeline.ocr_qwen import run_qwen_ocr
from app.pipeline.ocr_yandex import run_yandex_ocr
from app.pipeline.ocr_google_vision import run_google_vision_ocr
from app.pipeline.ocr_match import evaluate_ocr_wine_id_help, normalize_ocr_text
from app.pipeline.exclusive_lexicon import (
    evaluate_exclusive_lexicon,
    _cupage_from_label_ocr,
)
from app.pipeline.xgb_match import (
    apply_xgb_fin,
    attach_xgb_to_candidate_items,
    collect_query_ocr_text,
    enrich_wine_dict_for_xgb,
    evaluate_label_text_hard_reject,
    list_ocr_texts_by_engine,
    resolve_final_ocr,
    score_candidates_skipping_hard_reject,
    score_reuse_previous_search_xgb,
)
from app.pipeline.crenc_match import (
    apply_crenc_fin,
    attach_crenc_to_candidate_items,
    score_candidates as score_crenc_candidates,
    score_candidates_server as score_crenc_candidates_server,
)
from app.pipeline.openai_txt_match import (
    attach_openai_txt_to_candidate_items,
    evaluate_openai_txt_match,
)
from app.pipeline.runtime_settings import (
    get_settings,
    score_band,
    _clamp01,
    _clamp099,
    _normalize_text_match_thresholds,
)
from app.pipeline.version import (
    ALGORITHM_NOTES,
    ALGORITHM_UPDATED_AT,
    ALGORITHM_VERSION,
)
from app.pipeline.bottle_orient import (
    DEFAULT_MIN_DEG as BOTTLE_ORIENT_MIN_DEG,
    transform_box_dict,
    upright_work_image,
)
from app.pipeline.yolo_detect import (
    crop_box,
    crop_box_mask_foreign_labels,
    detect_bottle_and_labels,
    detect_label_boxes,
    draw_boxes,
    filter_bottles_by_min_conf,
    normalize_yolo_variant,
    select_primary_bottle,
    select_primary_bottle_with_labels,
    select_primary_label,
    select_primary_label_with_bottles,
)

_SAFE_NAME = re.compile(r"[^\w.\-()+ ]+", re.UNICODE)
_log = logging.getLogger("findwine")


def _plog(msg: str, **kv: Any) -> None:
    """Строка шага пайплайна в терминал (uvicorn)."""
    parts = [f"[findwine] {msg}"]
    for k, v in kv.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    line = " | ".join(parts)
    print(line, flush=True)
    _log.info(line)


def _safe_filename(name: str) -> str:
    base = Path(name or "upload.bin").name
    cleaned = _SAFE_NAME.sub("_", base).strip(" ._")
    return cleaned or "upload.bin"


def _dt_stamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone()
    return local.strftime("%Y%m%d-%H%M%S")


def _with_postfix(path: Path, postfix: str) -> Path:
    return path.with_name(f"{path.stem}{postfix}{path.suffix}")


def _work_jpeg_path(saved_path: Path) -> Path:
    """Normalized work JPEG path — never reuse the original file path.

    If the upload is already ``.jpg``/``.jpeg``, writing work with ``with_suffix('.jpg')``
    would overwrite the original bytes (and previously could leave a mislabeled WebP).
    """
    if saved_path.suffix.lower() in {".jpg", ".jpeg"}:
        return saved_path.with_name(f"{saved_path.stem}_work.jpg")
    return saved_path.with_suffix(".jpg")


def _json_safe(obj: Any) -> Any:
    """Make status JSON-serializable (numpy scalars from OCR/YOLO break jsonb)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        # Skip private in-process keys (e.g. candidate ``_hsv`` blob).
        return {
            str(k): _json_safe(v)
            for k, v in obj.items()
            if not str(k).startswith("_")
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return None
    # numpy / torch-ish scalars
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _json_safe(obj.item())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return _json_safe(obj.tolist())
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def _save_status(
    db: Session,
    row: SearchPhoto,
    status: dict[str, Any],
    *,
    commit: bool = False,
) -> None:
    """Persist status JSON to search_photos.

    During the pipeline status lives in memory only (commit=False no-op).
    Commit once on early abort or at the end of the run — avoids N round-trips
    that used to cost ~100ms+ each.
    """
    if not commit:
        return
    from app.scan_history_denorm import sync_search_photo_history_columns

    safe = _json_safe(status)
    if isinstance(safe, dict):
        status.clear()
        status.update(safe)
    row.status = status
    sync_search_photo_history_columns(row, status)
    if row.hist_wine_id is not None and not row.hist_wine_slug:
        wine = db.get(Wine, int(row.hist_wine_id))
        if wine is not None and wine.slug:
            row.hist_wine_slug = str(wine.slug).strip() or None
    db.add(row)
    db.commit()
    db.refresh(row)


def _enrich_candidates(
    db: Session, hits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach wine card fields to cosine hits (keep order)."""
    if not hits:
        return []
    ids = [h["id"] for h in hits]
    wines = {
        w.id: w
        for w in db.scalars(select(Wine).where(Wine.id.in_(ids))).all()
    }
    slugs = [w.slug for w in wines.values() if w.slug]
    sitemap_map: dict[str, str] = {}
    if slugs:
        rows = db.execute(
            select(WineSitemap.slug, WineSitemap.image_file).where(
                WineSitemap.slug.in_(slugs),
                WineSitemap.image_file.is_not(None),
            )
        ).all()
        sitemap_map = {
            str(slug): str(image_file)
            for slug, image_file in rows
            if image_file
        }

    out: list[dict[str, Any]] = []
    for h in hits:
        wine = wines.get(h["id"])
        if wine is None:
            out.append({**h, "name": None, "winery": None})
            continue
        sitemap_file = sitemap_map.get(wine.slug) if wine.slug else None
        lo = wine.label_ocr if isinstance(wine.label_ocr, dict) else {}
        type_canon = (
            _analog_type_canon(str(lo.get("type") or ""))
            or _analog_type_canon(str(wine.name or ""))
            or _analog_type_canon(str(wine.label or ""))
        )
        wine_type = (
            TYPE_LABELS.get(type_canon, type_canon) if type_canon else None
        )
        item: dict[str, Any] = {
            **h,
            "name": wine.name,
            "winery": wine.winery,
            "slug": wine.slug,
            "label": wine.label,
            "color": wine.color,
            "category": wine.category,
            "grape_variety": wine.grape_variety,
            "description": wine.description,
            "wine_type": wine_type,
            "region": wine.region,
            "photo_url": photo_url(
                wine.photo_name,
                slug=wine.slug,
                sitemap_image_file=sitemap_file,
            ),
            "label_url": label_url(wine.photo_name, slug=wine.slug),
        }
        # Reuse-hit: keep catalog label separately; UI/XGB use previous OCR.
        prev_ocr = str(h.get("previous_search_ocr") or "").strip()
        if h.get("from_previous_search") and prev_ocr:
            item["catalog_label"] = wine.label
            item["previous_search_ocr"] = prev_ocr
            item["label"] = prev_ocr
        # In-process only (stripped by _json_safe): catalog HSV hist from wines.hsv
        if wine.hsv:
            item["_hsv"] = bytes(wine.hsv)
        out.append(item)
    return out


def _run_cosine_search(
    db: Session,
    *,
    search_photos_id: int,
    emb_step: dict[str, Any],
    reuse_previous_searches: bool = False,
    candidates_top_n: int = 40,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reuse_meta: dict[str, Any] | None = None
    top_n = int(candidates_top_n) if int(candidates_top_n) in {10, 20, 30, 40} else 40
    for etype in ("siglip2", "dinov3"):
        if not emb_step.get(etype, {}).get("ok"):
            result[etype] = {"ok": False, "ids": [], "items": [], "error": "no embedding"}
            continue
        try:
            packed = search_top_wines_by_cosine(
                db,
                search_photos_id=search_photos_id,
                embedding_type=etype,
                limit=top_n,
                reuse_previous_searches=bool(reuse_previous_searches)
                and etype == "siglip2",
            )
            hits = list(packed.get("hits") or [])
            if etype == "siglip2" and packed.get("reuse"):
                reuse_meta = dict(packed["reuse"])
            items = _enrich_candidates(db, hits)
            result[etype] = {
                "ok": True,
                "ids": [h["id"] for h in hits],
                "items": items,
            }
        except Exception as exc:  # noqa: BLE001
            result[etype] = {
                "ok": False,
                "ids": [],
                "items": [],
                "error": str(exc),
            }
    if reuse_meta is not None:
        result["reuse_previous_search"] = reuse_meta
    return result


def _insert_embedding(
    db: Session,
    *,
    search_photos_id: int,
    embedding_type: str,
    embedding: list[float],
) -> int:
    if len(embedding) != EMBED_DIM:
        raise ValueError(f"ожидался dim={EMBED_DIM}, получили {len(embedding)}")
    literal = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    result = db.execute(
        text(
            """
            INSERT INTO search_photo_embeddings
                (search_photos_id, embedding_type, embedding)
            VALUES
                (:sid, :etype, CAST(:emb AS vector))
            ON CONFLICT (search_photos_id, embedding_type)
            DO UPDATE SET embedding = EXCLUDED.embedding
            RETURNING id
            """
        ),
        {"sid": search_photos_id, "etype": embedding_type, "emb": literal},
    )
    emb_id = int(result.scalar_one())
    # Flush only — commit together with final search_photos.status.
    db.flush()
    return emb_id


def run_findwine(
    db: Session,
    *,
    data: bytes,
    filename: str,
    eval_mode: int = 0,
) -> dict[str, Any]:
    """Full synchronous pipeline. Returns status JSON (also stored in DB).

    Победитель в eval_mode=0 и eval_mode=1 выбирается одинаково (только полоса
    match). Разница: eval_mode=1 (для /v1/eval/predict) не считает аналоги.

    search_photos.status пишется в БД один раз в конце (или при раннем abort);
    промежуточные шаги держат status только в памяти.
    """
    t_total = time.perf_counter()
    eval_flag = 1 if int(eval_mode) == 1 else 0
    # Расширение по содержимому файла, не по имени от клиента (часто .jpg на WebP)
    aligned_name, sniffed_ext = align_filename_extension(filename, data)
    original_name = _safe_filename(aligned_name)
    sha256 = hashlib.sha256(data).hexdigest()
    filesize = len(data)

    status: dict[str, Any] = {
        "ok": False,
        "error": None,
        "steps": {},
        "timings_ms": {},
        # Offset from pipeline start (ms) — for Gantt / parallel timeline
        "timings_start_ms": {},
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_updated_at": ALGORITHM_UPDATED_AT,
        "algorithm_notes": ALGORITHM_NOTES,
        # 0 — обычный findwine; 1 — запрос eval-скрипта. Не путать со status["eval"]
        # (ручные флаги FP/FN в истории сканов). Match-правила те же.
        "eval_mode": eval_flag,
    }

    def _time_set(
        key: str,
        t0: float,
        *,
        duration_ms: float | None = None,
        skip: bool = False,
    ) -> None:
        """Record start offset + duration; skip zero/disabled steps."""
        if skip:
            return
        if duration_ms is None:
            duration_ms = (time.perf_counter() - t0) * 1000
        try:
            dur = float(duration_ms)
        except (TypeError, ValueError):
            return
        if dur <= 0:
            return
        status["timings_start_ms"][key] = round((t0 - t_total) * 1000, 1)
        status["timings_ms"][key] = round(dur, 1)
        _plog(f"OK {key}", ms=status["timings_ms"][key])

    def _time_total() -> None:
        status["timings_start_ms"]["total"] = 0.0
        status["timings_ms"]["total"] = round(
            (time.perf_counter() - t_total) * 1000, 1
        )
    pipeline_settings = get_settings()
    status["settings"] = {
        "ocr_preprocess": list(pipeline_settings["ocr_preprocess"]),
        "ocr_engines": list(pipeline_settings.get("ocr_engines") or []),
        "text_weights": dict(pipeline_settings["text_weights"]),
        "final_weights": dict(pipeline_settings.get("final_weights") or {}),
        "use_dinov3": bool(pipeline_settings.get("use_dinov3", True)),
        "geometry_siglip2": bool(pipeline_settings.get("geometry_siglip2", True)),
        "geometry_dinov3": bool(pipeline_settings.get("geometry_dinov3", True)),
        "use_gemini_ocr": bool(pipeline_settings.get("use_gemini_ocr", True)),
        "use_openai_ocr": bool(pipeline_settings.get("use_openai_ocr", True)),
        "use_deepseek_ocr": bool(pipeline_settings.get("use_deepseek_ocr", True)),
        "use_qwen_ocr": bool(pipeline_settings.get("use_qwen_ocr", False)),
        "use_yandex_ocr": bool(pipeline_settings.get("use_yandex_ocr", True)),
        "use_google_vision_ocr": bool(
            pipeline_settings.get("use_google_vision_ocr", True)
        ),
        "yolo_fallback_full_image": bool(
            pipeline_settings.get("yolo_fallback_full_image", True)
        ),
        "use_yolo": bool(pipeline_settings.get("use_yolo", True)),
        "yolo_variant": normalize_yolo_variant(
            pipeline_settings.get("yolo_variant")
        ),
        "label_detect_openai": bool(
            pipeline_settings.get("label_detect_openai", True)
        ),
        "label_detect_gemini": bool(
            pipeline_settings.get("label_detect_gemini", True)
        ),
        "embedding_device": (
            "gpu"
            if str(pipeline_settings.get("embedding_device") or "cpu").lower()
            == "gpu"
            else "cpu"
        ),
        "embed_cpu_use_cache": bool(
            pipeline_settings.get("embed_cpu_use_cache", False)
        ),
        "normalize_max_side": int(
            pipeline_settings.get("normalize_max_side") or 1024
        ),
        "candidates_top_n": int(
            pipeline_settings.get("candidates_top_n") or 40
        ),
        "hf_start_timeout_sec": float(
            pipeline_settings.get("hf_start_timeout_sec") or 15
        ),
        "text_match_methods": list(
            pipeline_settings.get("text_match_methods") or []
        ),
        "final_score_method": str(
            pipeline_settings.get("final_score_method") or "fin1"
        ),
        "text_match_thresholds": _normalize_text_match_thresholds(
            pipeline_settings.get("text_match_thresholds")
        ),
        "empty_ocr_cosine_threshold": _clamp099(
            pipeline_settings.get("empty_ocr_cosine_threshold"), 0.86
        ),
        "compute_hsv": bool(pipeline_settings.get("compute_hsv", False)),
        "use_hsv_filter": bool(pipeline_settings.get("use_hsv_filter", False)),
        "hsv_bhattacharyya_max": _clamp01(
            pipeline_settings.get("hsv_bhattacharyya_max"), 0.50
        ),
        "hsv_ignore_high_cosine": bool(
            pipeline_settings.get("hsv_ignore_high_cosine", False)
        ),
        "hsv_ignore_cosine_min": _clamp099(
            pipeline_settings.get("hsv_ignore_cosine_min"), 0.85
        ),
        "use_hsv_hard_reject": bool(
            pipeline_settings.get("use_hsv_hard_reject", False)
        ),
        "hsv_hard_reject_max": _clamp01(
            pipeline_settings.get("hsv_hard_reject_max"), 0.90
        ),
        "hard_reject_ignore_high_scores": bool(
            pipeline_settings.get("hard_reject_ignore_high_scores", True)
        ),
        "hard_reject_ignore_cosine_min": _clamp099(
            pipeline_settings.get("hard_reject_ignore_cosine_min"), 0.90
        ),
        "hard_reject_ignore_xgb_min": _clamp01(
            pipeline_settings.get("hard_reject_ignore_xgb_min"), 0.70
        ),
        "compute_color_delta": bool(
            pipeline_settings.get("compute_color_delta", True)
        ),
        "reuse_previous_searches": bool(
            pipeline_settings.get("reuse_previous_searches", False)
        ),
        "show_search_details": bool(
            pipeline_settings.get("show_search_details", True)
        ),
        "final_ocr": str(pipeline_settings.get("final_ocr") or "auto"),
    }
    # Explicit alias for weights (requested field for search_photos.status)
    status["text_weights"] = dict(pipeline_settings["text_weights"])
    status["final_weights"] = dict(pipeline_settings.get("final_weights") or {})
    use_dinov3 = bool(pipeline_settings.get("use_dinov3", True))
    geometry_siglip2 = bool(pipeline_settings.get("geometry_siglip2", True))
    geometry_dinov3 = bool(pipeline_settings.get("geometry_dinov3", True))
    yolo_fallback_full_image = bool(
        pipeline_settings.get("yolo_fallback_full_image", True)
    )
    use_yolo = bool(pipeline_settings.get("use_yolo", True))
    yolo_variant = normalize_yolo_variant(pipeline_settings.get("yolo_variant"))
    bottle_min_conf = _clamp01(pipeline_settings.get("bottle_min_conf"), 0.60)
    use_bottle_orient = bool(pipeline_settings.get("use_bottle_orient", True))
    try:
        bottle_orient_min_deg = float(
            pipeline_settings.get("bottle_orient_min_deg", BOTTLE_ORIENT_MIN_DEG)
        )
    except (TypeError, ValueError):
        bottle_orient_min_deg = float(BOTTLE_ORIENT_MIN_DEG)
    bottle_orient_min_deg = max(1.0, min(45.0, bottle_orient_min_deg))
    label_detect_openai = bool(pipeline_settings.get("label_detect_openai", True))
    label_detect_gemini = bool(pipeline_settings.get("label_detect_gemini", True))
    use_gemini_ocr = bool(pipeline_settings.get("use_gemini_ocr", True))
    use_openai_ocr = bool(pipeline_settings.get("use_openai_ocr", True))
    use_deepseek_ocr = bool(pipeline_settings.get("use_deepseek_ocr", True))
    use_qwen_ocr = bool(pipeline_settings.get("use_qwen_ocr", False))
    use_yandex_ocr = bool(pipeline_settings.get("use_yandex_ocr", True))
    use_google_vision_ocr = bool(
        pipeline_settings.get("use_google_vision_ocr", True)
    )
    compute_hsv = bool(pipeline_settings.get("compute_hsv", False))
    use_hsv_filter = bool(pipeline_settings.get("use_hsv_filter", False)) and compute_hsv
    hsv_bhattacharyya_max = _clamp01(
        pipeline_settings.get("hsv_bhattacharyya_max"), 0.50
    )
    hsv_ignore_high_cosine = (
        bool(pipeline_settings.get("hsv_ignore_high_cosine", False)) and compute_hsv
    )
    hsv_ignore_cosine_min = _clamp099(
        pipeline_settings.get("hsv_ignore_cosine_min"), 0.85
    )
    use_hsv_hard_reject = (
        bool(pipeline_settings.get("use_hsv_hard_reject", False)) and compute_hsv
    )
    hsv_hard_reject_max = _clamp01(
        pipeline_settings.get("hsv_hard_reject_max"), 0.90
    )
    hard_reject_ignore_high_scores = bool(
        pipeline_settings.get("hard_reject_ignore_high_scores", True)
    )
    hard_reject_ignore_cosine_min = _clamp099(
        pipeline_settings.get("hard_reject_ignore_cosine_min"), 0.90
    )
    hard_reject_ignore_xgb_min = _clamp01(
        pipeline_settings.get("hard_reject_ignore_xgb_min"), 0.70
    )
    compute_color_delta = bool(
        pipeline_settings.get("compute_color_delta", True)
    )
    reuse_previous_searches = bool(
        pipeline_settings.get("reuse_previous_searches", False)
    )
    try:
        candidates_top_n = int(pipeline_settings.get("candidates_top_n") or 40)
    except (TypeError, ValueError):
        candidates_top_n = 40
    if candidates_top_n not in {10, 20, 30, 40}:
        candidates_top_n = 40
    final_ocr_pref = str(pipeline_settings.get("final_ocr") or "auto")
    try:
        normalize_max_side = int(pipeline_settings.get("normalize_max_side") or 1024)
    except (TypeError, ValueError):
        normalize_max_side = 1024
    if normalize_max_side not in (1280, 1024, 800):
        normalize_max_side = 1024
    embedding_device = (
        "gpu"
        if str(pipeline_settings.get("embedding_device") or "cpu").lower() == "gpu"
        else "cpu"
    )
    try:
        hf_start_timeout_sec = float(
            pipeline_settings.get("hf_start_timeout_sec") or 15
        )
    except (TypeError, ValueError):
        hf_start_timeout_sec = 15.0
    hf_start_timeout_sec = max(1.0, min(300.0, hf_start_timeout_sec))
    openai_from_detect: dict[str, Any] | None = None
    gemini_from_detect: dict[str, Any] | None = None
    openai_label_detect_ran = False
    gemini_label_detect_ran = False
    boxes: list[dict[str, Any]] = []
    label_box_source = "yolo" if use_yolo else "llm"

    _plog(
        "START pipeline start",
        file=original_name,
        bytes=filesize,
        algo=ALGORITHM_VERSION,
        yolo=use_yolo,
        embed=embedding_device,
        dinov3=use_dinov3,
        max_side=normalize_max_side,
    )

    # GPU embed: сначала прямой вызов HF (без resume). При pause —
    # фоновый start (для следующего скана) + CPU embed для текущего.
    # Qwen OCR: та же схема (pause → kick start, OCR в этом прогоне skip).
    want_gpu_embed = embedding_device == "gpu"
    need_hf_qwen = bool(use_qwen_ocr) and bool(
        (getattr(app_config, "HF_QWEN_OCR_ENDPOINT", "") or "").strip()
    )

    hf_sig_step: dict[str, Any] | None = None
    hf_dino_step: dict[str, Any] | None = None
    hf_qwen_step: dict[str, Any] | None = None
    _hf_kick_started: set[str] = set()
    _hf_kick_lock = threading.Lock()
    photo_id: int | None = None
    # Фон: resume может занять дольше UI-таймаута — для «следующего раза»
    hf_bg_timeout_sec = max(float(hf_start_timeout_sec), 180.0)

    if want_gpu_embed or need_hf_qwen:
        _plog(
            ">> hf_endpoint: try call first (no pre-start)",
            embed=embedding_device,
            qwen=need_hf_qwen,
        )
        # Keep-alive Session ready before first SigLIP HF probe
        if want_gpu_embed:
            get_hf_session()
            hf_url = (
                getattr(app_config, "HF_SIGLIP2_ENDPOINT", "")
                or getattr(app_config, "HF_ENDPOINT", "")
                or ""
            ).strip()
            if hf_url:
                warm = warm_hf_connection(hf_url, timeout=min(3.0, hf_start_timeout_sec))
                status["steps"]["hf_connection_warm"] = warm
                _plog(
                    "hf_connection_warm",
                    ok=warm.get("ok"),
                    ms=warm.get("ms"),
                    error=(warm.get("error") or "")[:120] or None,
                )
    else:
        _plog(">> hf_endpoint skipped", reason="cpu embed and qwen off")

    def _shutdown_hf_start_pool(*, force: bool = False) -> None:
        # Фоновые kick-start живут в daemon-потоках — не отменяем
        # (нужны для следующего скана). force оставлен для call-sites.
        return

    def _persist_hf_start_to_db(
        *,
        timing_key: str,
        step: dict[str, Any],
        start_ms: float | None,
        duration_ms: float | None,
    ) -> None:
        """No-op: HF steps stay in-memory; final _save_status(commit=True) writes them.

        Earlier this opened a second Session and merged into search_photos.status
        mid-run — extra commits during the scan. Status is already updated under
        _hf_kick_lock in _record_hf_start_step.
        """
        return

    def _record_hf_start_step(role: str, step: dict[str, Any], t0: float) -> None:
        nonlocal hf_sig_step, hf_dino_step, hf_qwen_step
        timing_key = f"hf_endpoint_start_{role}"
        start_ms: float | None = None
        duration_ms: float | None = None
        with _hf_kick_lock:
            if role == "siglip2":
                hf_sig_step = step
            elif role == "dinov3":
                hf_dino_step = step
            else:
                hf_qwen_step = step
            status["steps"][timing_key] = step
            hf_ms = step.get("ms")
            if hf_ms and float(hf_ms) > 0:
                _time_set(timing_key, t0, duration_ms=hf_ms)
                start_ms = status["timings_start_ms"].get(timing_key)
                duration_ms = status["timings_ms"].get(timing_key)
            if step.get("ok"):
                _plog(f"OK {timing_key} ready (bg)", ok=True, ms=step.get("ms"))
            else:
                status["steps"][timing_key]["note"] = (
                    step.get("error")
                    or f"HF {role} start не завершился — для следующего скана"
                )
                _plog(
                    f"FAIL {timing_key} (bg)",
                    ok=False,
                    error=step.get("error"),
                    ms=step.get("ms"),
                )
        # photo_id появляется после INSERT — kick только после label_crop
        if photo_id:
            _persist_hf_start_to_db(
                timing_key=timing_key,
                step=step,
                start_ms=start_ms,
                duration_ms=duration_ms,
            )

    def _kick_hf_start(role: str, *, t_kick: float | None = None) -> float:
        """Resume HF в daemon-потоке. Не ждём. Возвращает t_kick для Gantt."""
        role = (role or "").strip().lower()
        if role not in ("siglip2", "dinov3", "qwen_ocr"):
            return time.perf_counter() if t_kick is None else t_kick
        with _hf_kick_lock:
            if role in _hf_kick_started:
                return t_kick if t_kick is not None else time.perf_counter()
            _hf_kick_started.add(role)
        if t_kick is None:
            t_kick = time.perf_counter()
        timing_key = f"hf_endpoint_start_{role}"
        # Сразу ставим pending, чтобы Timeline показал старт параллельно с CPU
        with _hf_kick_lock:
            status["steps"][timing_key] = {
                "ok": None,
                "pending": True,
                "role": role,
                "provider": "huggingface",
                "action": "start",
                "deferred": True,
                "note": "resume в фоне (без wait); текущий скан → CPU/local",
            }
        _plog(f">> hf_endpoint_start_{role} (bg resume, no wait; parallel CPU)")

        def _bg() -> None:
            try:
                # Не ждём готовности endpoint — только resume в фоне.
                # Текущий скан уже ушёл на CPU / local.
                step = start_hf_endpoint(
                    role, timeout_sec=hf_bg_timeout_sec, wait=False
                )
            except Exception as exc:  # noqa: BLE001
                step = {
                    "ok": False,
                    "role": role,
                    "error": str(exc),
                    "provider": "huggingface",
                    "action": "start",
                    "waited": False,
                    "ms": round((time.perf_counter() - t_kick) * 1000, 1),
                }
            step["deferred"] = True
            step["pending"] = False
            step["waited"] = bool(step.get("waited", False))
            step["note"] = step.get("note") or (
                "resume в фоне без ожидания; текущий embed → CPU/local"
            )
            # Длительность от момента kick (только API resume, без wait)
            step["ms"] = round((time.perf_counter() - t_kick) * 1000, 1)
            try:
                _record_hf_start_step(role, step, t_kick)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(
            target=_bg, name=f"hf-kick-{role}", daemon=True
        ).start()
        return t_kick

    def _flush_hf_start_status() -> None:
        """Записать уже готовые bg-step'ы (без ожидания)."""
        with _hf_kick_lock:
            for role, step in (
                ("siglip2", hf_sig_step),
                ("dinov3", hf_dino_step),
                ("qwen_ocr", hf_qwen_step),
            ):
                if step is None:
                    continue
                key = f"hf_endpoint_start_{role}"
                if key not in status["steps"]:
                    status["steps"][key] = step

    def _embed_gpu_then_cpu(
        *,
        role: str,
        embed_fn: Any,
    ) -> dict[str, Any]:
        """GPU без pre-start; при pause — start∥CPU, итог = CPU (не ждём start)."""
        # Короткая проба GPU: не блокировать CPU на полном HF_TIMEOUT
        gpu_probe_sec = min(
            8.0,
            max(2.0, float(hf_start_timeout_sec)),
        )
        t_gpu = time.perf_counter()
        try:
            vec, meta = embed_fn(
                label_path,
                force_cpu=False,
                device="gpu",
                timeout=gpu_probe_sec,
            )
            device = str((meta or {}).get("provider") or "gpu")
            if device not in ("cpu", "gpu"):
                device = "gpu"
            return {
                "ok": True,
                "vec": vec,
                "meta": {**(meta or {}), "device": device, "provider": device},
                "ms": round((time.perf_counter() - t_gpu) * 1000, 1),
                "t_start": t_gpu,
                "device": device,
            }
        except Exception as gpu_exc:  # noqa: BLE001
            paused = is_hf_endpoint_paused_error(gpu_exc)
            network = is_hf_endpoint_network_error(gpu_exc)
            t_fail = time.perf_counter()
            # Resume HF in background for next scan (pause OR SSL/network blip)
            if paused or network:
                _kick_hf_start(role, t_kick=t_fail)
                _plog(
                    f"HF {role} {'paused' if paused else 'network'} >> start∥CPU",
                    error=str(gpu_exc)[:200],
                )
            else:
                _plog(
                    f"HF {role} failed >> CPU embed",
                    error=str(gpu_exc)[:200],
                )

            # CPU сразу в этом же потоке; HF start уже в daemon (если kick)
            t_cpu = time.perf_counter()
            try:
                vec, meta = embed_fn(
                    label_path, force_cpu=True, device="gpu"
                )
                meta = {
                    **(meta or {}),
                    "device": "cpu",
                    "provider": "cpu",
                    "fallback_from_gpu": True,
                    "requested_device": "gpu",
                    "gpu_error": str(gpu_exc)[:400],
                    "hf_paused": paused,
                    "hf_network_error": network,
                    "hf_start_kicked": bool(paused or network),
                    "gpu_probe_timeout_sec": gpu_probe_sec,
                }
                return {
                    "ok": True,
                    "vec": vec,
                    "meta": meta,
                    "ms": round((time.perf_counter() - t_cpu) * 1000, 1),
                    "t_start": t_cpu,
                    "device": "cpu",
                    "gpu_attempt_ms": round((t_fail - t_gpu) * 1000, 1),
                }
            except Exception as cpu_exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "error": (
                        f"GPU failed ({str(gpu_exc)[:180]}); "
                        f"CPU fallback failed ({str(cpu_exc)[:220]})"
                    ),
                    "gpu_error": str(gpu_exc),
                    "ms": round((time.perf_counter() - t_cpu) * 1000, 1),
                    "t_start": t_cpu,
                    "device": "cpu",
                }

    row = SearchPhoto(
        filename=original_name,
        filesize=filesize,
        sha256=sha256,
        status=status,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    photo_id = row.id
    created_at = row.created_at or datetime.now(timezone.utc)
    stamp = _dt_stamp(created_at)

    SEARCH_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    saved_name = f"{photo_id}_{stamp}_{original_name}"
    saved_path = SEARCH_PHOTO_DIR / saved_name

    _plog(">> save", id=photo_id, filename=saved_name)
    t0 = time.perf_counter()
    saved_path.write_bytes(data)
    status["steps"]["save"] = {
        "path": str(saved_path),
        "filename": saved_name,
        "filesize": filesize,
        "sha256": sha256,
        "client_filename": _safe_filename(filename),
        "content_ext": sniffed_ext,
    }
    _time_set("save", t0)
    status["search_photos_id"] = photo_id
    status["created_at"] = created_at.isoformat()
    _save_status(db, row, status)

    # Normalize to RGB JPEG for YOLO / OpenCV / OCR (HEIC and others).
    # Invalid images fail here — отдельный validate не нужен.
    _plog(">> normalize", max_side=normalize_max_side)
    t0 = time.perf_counter()
    try:
        rgb, norm_meta = normalize_image_bytes(data, max_side=normalize_max_side)
        work_path = _work_jpeg_path(saved_path)
        budget_meta = save_work_jpeg_budget(
            rgb,
            work_path,
            max_bytes=WORK_JPEG_MAX_BYTES,
            max_side=normalize_max_side,
            quality_start=NORMALIZE_JPEG_QUALITY,
        )
        status["steps"]["normalize"] = {
            "ok": True,
            "work_path": str(work_path),
            "work_filename": work_path.name,
            "work_bytes": budget_meta.get("bytes"),
            "work_max_bytes": WORK_JPEG_MAX_BYTES,
            "source_format": norm_meta.get("source_format"),
            "mode": norm_meta.get("mode"),
            "normalize_max_side": normalize_max_side,
            "original_size": norm_meta.get("original_size"),
            "size": budget_meta.get("size") or norm_meta.get("size"),
            "resized": bool(norm_meta.get("resized"))
            or bool(budget_meta.get("shrunk_for_budget")),
            "scale": norm_meta.get("scale"),
            "jpeg_quality": budget_meta.get("jpeg_quality") or NORMALIZE_JPEG_QUALITY,
            "decode_backend": norm_meta.get("decode_backend"),
            "resize_backend": norm_meta.get("resize_backend"),
            "exif_orientation": norm_meta.get("exif_orientation"),
            "shrunk_for_budget": bool(budget_meta.get("shrunk_for_budget")),
            "budget_miss": bool(budget_meta.get("budget_miss")),
        }
        status["work_filename"] = work_path.name
        status["normalize_max_side"] = normalize_max_side
    except Exception as exc:  # noqa: BLE001
        status["steps"]["normalize"] = {
            "ok": False,
            "error": str(exc),
            "normalize_max_side": normalize_max_side,
        }
        status["ok"] = False
        status["error"] = f"normalize: {exc}"
        _plog("FAIL normalize failed", error=str(exc))
        _time_set("normalize", t0)
        _time_total()
        _save_status(db, row, status, commit=True)
        return status
    _plog(
        "normalize size",
        original=status["steps"]["normalize"].get("original_size"),
        size=status["steps"]["normalize"].get("size"),
        resized=status["steps"]["normalize"].get("resized"),
        backend=status["steps"]["normalize"].get("decode_backend"),
        resize=status["steps"]["normalize"].get("resize_backend"),
    )
    _time_set("normalize", t0)
    _save_status(db, row, status)

    # 1) Label box: YOLO or LLM detect+OCR (OpenAI / Gemini) when YOLO disabled.
    # HF start уже бежит с начала пайплайна — здесь его не ждём.
    _plog(
        ">> label_detect",
        mode="yolo" if use_yolo else "llm",
        yolo_variant=yolo_variant if use_yolo else None,
        openai=label_detect_openai,
        gemini=label_detect_gemini,
    )
    t0 = time.perf_counter()
    crops_path = _with_postfix(work_path, "_crops")
    label_path = _with_postfix(work_path, "_label")
    label_hf_path = label_path  # overwritten after crop with short-side 384 JPEG
    selected_box: dict[str, Any] | None = None
    bottles_for_draw: list[dict[str, Any]] = []
    labels_for_draw: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []

    if use_yolo:
        try:
            if yolo_variant == "bottle_label":
                det = detect_bottle_and_labels(work_path)
                boxes = det["label_boxes"]
                bottles_for_draw = list(det.get("bottles") or [])
                labels_for_draw = list(det.get("labels") or [])
                labels_ignored = list(det.get("labels_ignored") or [])
                selected_box = select_primary_label_with_bottles(
                    bottles_for_draw,
                    boxes,
                    bottle_min_conf=bottle_min_conf,
                )
                draw_boxes(
                    work_path,
                    labels_for_draw,
                    crops_path,
                    thickness=5,
                    primary=selected_box,
                    bottles=bottles_for_draw,
                )
            else:
                boxes = detect_label_boxes(work_path, variant="label")
                labels_for_draw = list(boxes)
                labels_ignored = []
                selected_box = select_primary_label(boxes)
                draw_boxes(
                    work_path,
                    boxes,
                    crops_path,
                    thickness=5,
                    primary=selected_box,
                )
            status["steps"]["yolo"] = {
                "ok": True,
                "variant": yolo_variant,
                "bottle_min_conf": bottle_min_conf,
                "boxes_count": len(boxes),
                "bottles_count": len(bottles_for_draw),
                "raw_labels_count": len(labels_for_draw),
                "ignored_labels_count": len(labels_ignored),
                "boxes": [
                    {
                        "xyxy": [round(v, 2) for v in b["xyxy"]],
                        "conf": round(float(b["conf"]), 4),
                        "cls_name": b["cls_name"],
                        "area": round(float(b["area"]), 1),
                        **(
                            {
                                "merged": True,
                                "merged_count": b.get("merged_count"),
                            }
                            if b.get("merged")
                            else {}
                        ),
                    }
                    for b in boxes
                ],
                "bottles": [
                    {
                        "xyxy": [round(v, 2) for v in b["xyxy"]],
                        "conf": round(float(b["conf"]), 4),
                        "cls_name": b["cls_name"],
                        "area": round(float(b["area"]), 1),
                    }
                    for b in bottles_for_draw
                ],
                "raw_labels": [
                    {
                        "xyxy": [round(v, 2) for v in b["xyxy"]],
                        "conf": round(float(b["conf"]), 4),
                        "cls_name": b["cls_name"],
                        "area": round(float(b["area"]), 1),
                    }
                    for b in labels_for_draw
                ],
                "ignored_labels": [
                    {
                        "xyxy": [round(v, 2) for v in b["xyxy"]],
                        "conf": round(float(b["conf"]), 4),
                        "cls_name": b["cls_name"],
                        "area": round(float(b["area"]), 1),
                        "ignore_reason": b.get("ignore_reason"),
                        "best_bottle_ratio": b.get("best_bottle_ratio"),
                    }
                    for b in labels_ignored
                ],
                "crops_path": str(crops_path),
                "crops_filename": crops_path.name,
                "selected": (
                    {
                        "xyxy": [round(v, 2) for v in selected_box["xyxy"]],
                        "conf": round(float(selected_box["conf"]), 4),
                        "cls_name": selected_box["cls_name"],
                        "area": round(float(selected_box["area"]), 1),
                        "select_reason": selected_box.get("select_reason"),
                        **(
                            {
                                "merged": True,
                                "merged_count": selected_box.get("merged_count"),
                            }
                            if selected_box.get("merged")
                            else {}
                        ),
                        **(
                            {"from_bottle": True}
                            if selected_box.get("from_bottle")
                            else {}
                        ),
                        **(
                            {"selection": selected_box.get("selection")}
                            if selected_box.get("selection")
                            else {}
                        ),
                        **(
                            {
                                "ignored_for_select": selected_box.get(
                                    "ignored_for_select"
                                )
                            }
                            if selected_box.get("ignored_for_select")
                            else {}
                        ),
                    }
                    if selected_box
                    else None
                ),
                "largest": (
                    {
                        "xyxy": [round(v, 2) for v in boxes[0]["xyxy"]],
                        "conf": round(float(boxes[0]["conf"]), 4),
                        "cls_name": boxes[0]["cls_name"],
                        "area": round(float(boxes[0]["area"]), 1),
                    }
                    if boxes
                    else None
                ),
            }
            status["crops_filename"] = crops_path.name
            if not selected_box:
                if yolo_fallback_full_image:
                    status["steps"]["yolo"]["fallback"] = "full_image"
                    status["steps"]["yolo"]["note"] = (
                        "этикетка не найдена — используем исходное фото"
                    )
                else:
                    status["ok"] = False
                    status["error"] = "YOLO не нашёл этикетку на изображении"
                    _plog("FAIL yolo: no selected box")
                    _time_set("yolo", t0)
                    _time_total()
                    _save_status(db, row, status, commit=True)
                    _shutdown_hf_start_pool(force=True)
                    return status
        except Exception as exc:  # noqa: BLE001
            status["steps"]["yolo"] = {"ok": False, "error": str(exc)}
            status["ok"] = False
            status["error"] = f"YOLO: {exc}"
            _plog("FAIL yolo failed", error=str(exc))
            _time_set("yolo", t0)
            _time_total()
            _save_status(db, row, status, commit=True)
            _shutdown_hf_start_pool(force=True)
            return status
        _plog(
            "yolo boxes",
            variant=yolo_variant,
            count=len(boxes),
            bottles=len(bottles_for_draw),
            selected=bool(selected_box),
            fallback=status["steps"]["yolo"].get("fallback"),
        )
        _time_set("yolo", t0)
        _save_status(db, row, status)
    else:
        # LLM: рамка этикетки + OCR одним промптом (вместо YOLO)
        if not label_detect_openai and not label_detect_gemini:
            status["steps"]["label_detect"] = {
                "ok": False,
                "error": "YOLO выкл. и не выбран ни OpenAI, ни Gemini для рамки",
            }
            if yolo_fallback_full_image:
                status["steps"]["label_detect"]["fallback"] = "full_image"
                status["steps"]["label_detect"]["note"] = (
                    "нет LLM detect — используем исходное фото"
                )
                boxes = []
                selected_box = None
                label_box_source = "full_image"
                _time_set("label_detect", t0)
            else:
                status["ok"] = False
                status["error"] = (
                    "YOLO выкл.: включите label_detect_openai и/или label_detect_gemini"
                )
                _time_set("label_detect", t0)
                _time_total()
                _save_status(db, row, status, commit=True)
                _shutdown_hf_start_pool(force=True)
                return status
        else:
            try:
                img_rgb, _ = open_image_rgb(work_path)
                w0, h0 = img_rgb.size
                detect_results: dict[str, dict[str, Any]] = {}

                def _run_openai_detect() -> dict[str, Any]:
                    return run_openai_label_detect_ocr(
                        work_path, image_size=(w0, h0)
                    )

                def _run_gemini_detect() -> dict[str, Any]:
                    return run_gemini_label_detect_ocr(
                        work_path, image_size=(w0, h0)
                    )

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=2
                ) as detect_pool:
                    futs: dict[str, concurrent.futures.Future] = {}
                    if label_detect_openai:
                        futs["openai"] = detect_pool.submit(_run_openai_detect)
                    if label_detect_gemini:
                        futs["gemini"] = detect_pool.submit(_run_gemini_detect)
                    for key, fut in futs.items():
                        try:
                            detect_results[key] = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            detect_results[key] = {
                                "ok": False,
                                "error": str(exc),
                                "boxes": [],
                                "selected": None,
                            }

                all_boxes: list[dict[str, Any]] = []
                sources_ok: list[str] = []
                for key, detect in detect_results.items():
                    step_id = f"{key}_label_detect"
                    status["steps"][step_id] = {
                        k: v
                        for k, v in detect.items()
                        if k not in {"boxes"}
                    }
                    d_boxes = list(detect.get("boxes") or [])
                    all_boxes.extend(d_boxes)
                    # Любой вызов label+OCR >> отдельный OCR-only этого LLM запрещён
                    if key == "openai":
                        openai_label_detect_ran = True
                        if detect.get("ocr_variant"):
                            openai_from_detect = detect["ocr_variant"]
                    elif key == "gemini":
                        gemini_label_detect_ran = True
                        if detect.get("ocr_variant"):
                            gemini_from_detect = detect["ocr_variant"]
                    if detect.get("ok") or d_boxes:
                        sources_ok.append(key)
                    d_ms = detect.get("ms")
                    if d_ms:
                        _time_set(step_id, t0, duration_ms=d_ms)

                boxes = all_boxes
                # Оба LLM detect >> кроп/_label/embedding по рамке Gemini
                prefer_gemini_box = bool(
                    label_detect_openai and label_detect_gemini
                )
                if prefer_gemini_box:
                    gemini_boxes = list(
                        (detect_results.get("gemini") or {}).get("boxes") or []
                    )
                    selected_box = select_primary_label(gemini_boxes)
                    if selected_box:
                        label_box_source = "gemini"
                        selected_box = {
                            **selected_box,
                            "select_reason": (
                                f"prefer_gemini:{selected_box.get('select_reason') or 'largest'}"
                            ),
                        }
                    else:
                        selected_box = select_primary_label(boxes)
                        if selected_box and selected_box.get("source"):
                            label_box_source = str(selected_box["source"])
                        elif sources_ok:
                            label_box_source = "+".join(sources_ok)
                        else:
                            label_box_source = "llm"
                else:
                    selected_box = select_primary_label(boxes)
                    if selected_box and selected_box.get("source"):
                        label_box_source = str(selected_box["source"])
                    elif sources_ok:
                        label_box_source = "+".join(sources_ok)
                    else:
                        label_box_source = "llm"

                draw_boxes(
                    work_path, boxes, crops_path, thickness=5, primary=selected_box
                )
                status["crops_filename"] = crops_path.name
                for key in detect_results:
                    step_id = f"{key}_label_detect"
                    if step_id in status["steps"]:
                        status["steps"][step_id]["crops_filename"] = crops_path.name
                        status["steps"][step_id]["boxes_count"] = len(
                            detect_results[key].get("boxes") or []
                        )

                status["steps"]["label_detect"] = {
                    "ok": bool(boxes),
                    "engines": list(detect_results.keys()),
                    "boxes_count": len(boxes),
                    "prefer_gemini_box": prefer_gemini_box,
                    "selected_source": (
                        selected_box.get("source") if selected_box else None
                    ),
                    "select_reason": (
                        selected_box.get("select_reason") if selected_box else None
                    ),
                    "crops_filename": crops_path.name,
                }

                if not boxes:
                    if yolo_fallback_full_image:
                        status["steps"]["label_detect"]["fallback"] = "full_image"
                        status["steps"]["label_detect"]["note"] = (
                            "этикетка не найдена — используем исходное фото"
                        )
                        label_box_source = "full_image"
                    else:
                        errs = [
                            f"{k}: {detect_results[k].get('error')}"
                            for k in detect_results
                            if detect_results[k].get("error")
                        ]
                        status["ok"] = False
                        status["error"] = (
                            "; ".join(errs)
                            if errs
                            else "LLM не нашёл этикетку на изображении"
                        )
                        _plog("FAIL label_detect: no boxes", error=status["error"])
                        _time_total()
                        _save_status(db, row, status, commit=True)
                        _shutdown_hf_start_pool(force=True)
                        return status
            except Exception as exc:  # noqa: BLE001
                status["steps"]["label_detect"] = {"ok": False, "error": str(exc)}
                status["ok"] = False
                status["error"] = f"LLM label detect: {exc}"
                _plog("FAIL label_detect failed", error=str(exc))
                _time_set("label_detect", t0)
                _time_total()
                _save_status(db, row, status, commit=True)
                _shutdown_hf_start_pool(force=True)
                return status
            ld = status["steps"].get("label_detect") or {}
            _plog(
                "label_detect boxes",
                count=ld.get("boxes_count"),
                source=ld.get("selected_source") or label_box_source,
                engines=ld.get("engines"),
                prefer_gemini=ld.get("prefer_gemini_box"),
            )
        _save_status(db, row, status)

    # 1b) выравнивание: оригинал work не трогаем; повёрнутый → *_orient.jpg + YOLO снова
    _plog(">> bottle_orient", enabled=use_bottle_orient)
    t_orient = time.perf_counter()
    pipeline_path = work_path  # кадр для crop / OCR / embed
    orient_step: dict[str, Any] = {
        "ok": True,
        "applied": False,
        "enabled": bool(use_bottle_orient),
        "min_deg": bottle_orient_min_deg,
        "reason": "disabled" if not use_bottle_orient else "no_bottle",
    }
    if use_bottle_orient:
        orient_xyxy: list[float] | None = None
        filtered = filter_bottles_by_min_conf(bottles_for_draw, bottle_min_conf)
        primary_bottle = select_primary_bottle(list(filtered.get("bottles") or []))
        if primary_bottle and primary_bottle.get("xyxy"):
            orient_xyxy = [float(v) for v in primary_bottle["xyxy"]]
            orient_step["bottle_conf"] = round(
                float(primary_bottle.get("conf") or 0), 4
            )
        elif selected_box and selected_box.get("xyxy"):
            orient_xyxy = [float(v) for v in selected_box["xyxy"]]
            orient_step["reason"] = "fallback_label_box"
        if orient_xyxy is not None:
            try:
                orient_path = work_path.with_name(
                    f"{work_path.stem}_orient{work_path.suffix}"
                )
                ores = upright_work_image(
                    work_path,
                    orient_xyxy,
                    out_path=orient_path,
                    min_deg=bottle_orient_min_deg,
                    jpeg_quality=NORMALIZE_JPEG_QUALITY,
                )
                orient_step.update(
                    {
                        k: ores.get(k)
                        for k in (
                            "ok",
                            "applied",
                            "tilt_deg",
                            "rotate_deg",
                            "flip_180",
                            "reason",
                            "method",
                            "lines_n",
                            "edge_px",
                            "roi",
                            "size_in",
                            "size_out",
                            "ms",
                            "residual_deg",
                            "scale",
                            "orient_filename",
                            "orient_path",
                        )
                        if k in ores
                    }
                )
                orient_step["bottle_xyxy"] = [round(v, 2) for v in orient_xyxy]
                if ores.get("applied") and orient_path.is_file():
                    pipeline_path = orient_path
                    # Повторный YOLO на выровненном кадре (рамки в координатах orient)
                    t_y2 = time.perf_counter()
                    labels_ignored: list[dict[str, Any]] = []
                    if use_yolo:
                        if yolo_variant == "bottle_label":
                            det2 = detect_bottle_and_labels(orient_path)
                            boxes = det2["label_boxes"]
                            bottles_for_draw = list(det2.get("bottles") or [])
                            labels_for_draw = list(det2.get("labels") or [])
                            labels_ignored = list(
                                det2.get("labels_ignored") or []
                            )
                            selected_box = select_primary_label_with_bottles(
                                bottles_for_draw,
                                boxes,
                                bottle_min_conf=bottle_min_conf,
                            )
                            draw_boxes(
                                orient_path,
                                labels_for_draw,
                                crops_path,
                                thickness=5,
                                primary=selected_box,
                                bottles=bottles_for_draw,
                            )
                        else:
                            boxes = detect_label_boxes(
                                orient_path, variant="label"
                            )
                            labels_for_draw = list(boxes)
                            bottles_for_draw = []
                            selected_box = select_primary_label(boxes)
                            draw_boxes(
                                orient_path,
                                boxes,
                                crops_path,
                                thickness=5,
                                primary=selected_box,
                            )
                        yolo_rerun_ms = round(
                            (time.perf_counter() - t_y2) * 1000, 1
                        )
                        orient_step["yolo_rerun_ms"] = yolo_rerun_ms
                        status["crops_filename"] = crops_path.name
                        status["steps"]["yolo"] = {
                            "ok": True,
                            "variant": yolo_variant,
                            "bottle_min_conf": bottle_min_conf,
                            "after_orient": True,
                            "boxes_count": len(boxes),
                            "bottles_count": len(bottles_for_draw),
                            "raw_labels_count": len(labels_for_draw),
                            "ignored_labels_count": len(labels_ignored),
                            "boxes": [
                                {
                                    "xyxy": [round(v, 2) for v in b["xyxy"]],
                                    "conf": round(float(b["conf"]), 4),
                                    "cls_name": b["cls_name"],
                                    "area": round(float(b["area"]), 1),
                                    **(
                                        {
                                            "merged": True,
                                            "merged_count": b.get(
                                                "merged_count"
                                            ),
                                        }
                                        if b.get("merged")
                                        else {}
                                    ),
                                }
                                for b in boxes
                            ],
                            "bottles": [
                                {
                                    "xyxy": [round(v, 2) for v in b["xyxy"]],
                                    "conf": round(float(b["conf"]), 4),
                                    "cls_name": b["cls_name"],
                                    "area": round(float(b["area"]), 1),
                                }
                                for b in bottles_for_draw
                            ],
                            "raw_labels": [
                                {
                                    "xyxy": [round(v, 2) for v in b["xyxy"]],
                                    "conf": round(float(b["conf"]), 4),
                                    "cls_name": b["cls_name"],
                                    "area": round(float(b["area"]), 1),
                                }
                                for b in labels_for_draw
                            ],
                            "ignored_labels": [
                                {
                                    "xyxy": [round(v, 2) for v in b["xyxy"]],
                                    "conf": round(float(b["conf"]), 4),
                                    "cls_name": b["cls_name"],
                                    "area": round(float(b["area"]), 1),
                                    "ignore_reason": b.get("ignore_reason"),
                                    "best_bottle_ratio": b.get(
                                        "best_bottle_ratio"
                                    ),
                                }
                                for b in labels_ignored
                            ],
                            "crops_path": str(crops_path),
                            "crops_filename": crops_path.name,
                            "source_image": orient_path.name,
                            "selected": (
                                {
                                    "xyxy": [
                                        round(v, 2)
                                        for v in selected_box["xyxy"]
                                    ],
                                    "conf": round(
                                        float(selected_box["conf"]), 4
                                    ),
                                    "cls_name": selected_box["cls_name"],
                                    "area": round(
                                        float(selected_box["area"]), 1
                                    ),
                                    "select_reason": selected_box.get(
                                        "select_reason"
                                    ),
                                    **(
                                        {
                                            "merged": True,
                                            "merged_count": selected_box.get(
                                                "merged_count"
                                            ),
                                        }
                                        if selected_box.get("merged")
                                        else {}
                                    ),
                                    **(
                                        {"from_bottle": True}
                                        if selected_box.get("from_bottle")
                                        else {}
                                    ),
                                    **(
                                        {
                                            "selection": selected_box.get(
                                                "selection"
                                            )
                                        }
                                        if selected_box.get("selection")
                                        else {}
                                    ),
                                }
                                if selected_box
                                else None
                            ),
                        }
                        if not selected_box and yolo_fallback_full_image:
                            status["steps"]["yolo"]["fallback"] = "full_image"
                            status["steps"]["yolo"]["note"] = (
                                "после orient этикетка не найдена — "
                                "используем выровненный кадр"
                            )
                    else:
                        # без YOLO: только повёрнутый кадр для crop; рамки
                        # переносим матрицей (грубо) на _crops
                        m = np.asarray(ores.get("matrix"), dtype=np.float64)
                        if m is not None and m.size:
                            bottles_for_draw = [
                                transform_box_dict(b, m)
                                for b in bottles_for_draw
                            ]
                            labels_for_draw = [
                                transform_box_dict(b, m)
                                for b in labels_for_draw
                            ]
                            boxes = [transform_box_dict(b, m) for b in boxes]
                            if selected_box:
                                selected_box = transform_box_dict(
                                    selected_box, m
                                )
                        try:
                            draw_boxes(
                                orient_path,
                                labels_for_draw or boxes,
                                crops_path,
                                thickness=5,
                                primary=selected_box,
                                bottles=bottles_for_draw,
                            )
                            status["crops_filename"] = crops_path.name
                        except Exception as draw_exc:  # noqa: BLE001
                            orient_step["redraw_error"] = str(draw_exc)
            except Exception as exc:  # noqa: BLE001
                orient_step["ok"] = False
                orient_step["reason"] = f"error:{exc}"
                _plog("FAIL bottle_orient", error=str(exc))
    status["steps"]["orient"] = orient_step
    status["pipeline_image"] = pipeline_path.name
    _plog(
        "bottle_orient",
        applied=orient_step.get("applied"),
        tilt=orient_step.get("tilt_deg"),
        reason=orient_step.get("reason"),
        method=orient_step.get("method"),
        pipeline=pipeline_path.name,
        yolo_rerun_ms=orient_step.get("yolo_rerun_ms"),
    )
    _time_set("orient", t_orient)
    _save_status(db, row, status)

    # 2) selected box >> _label; if empty + fallback >> full pipeline image as label
    _plog(">> label_crop", source=label_box_source, has_box=bool(selected_box))
    t0 = time.perf_counter()
    try:
        if selected_box and selected_box.get("xyxy"):
            yolo_live = status["steps"].get("yolo") or {}
            bottles_live = (
                yolo_live.get("bottles")
                if isinstance(yolo_live, dict)
                else None
            ) or []
            raw_labels_live = (
                yolo_live.get("raw_labels")
                if isinstance(yolo_live, dict)
                else None
            ) or []
            foreign_mask: dict[str, Any] | None = None
            if bottles_live and raw_labels_live:
                # Primary = бутылка выбранной этикетки (не max-area), иначе
                # foreign_mask считает selected «чужой» и заливает весь кроп.
                primary_for_mask, _ = select_primary_bottle_with_labels(
                    list(bottles_live),
                    list(raw_labels_live),
                )
                foreign_mask = crop_box_mask_foreign_labels(
                    pipeline_path,
                    selected_box["xyxy"],
                    label_path,
                    bottles=list(bottles_live),
                    raw_labels=list(raw_labels_live),
                    primary_bottle=primary_for_mask,
                )
            else:
                crop_box(pipeline_path, selected_box["xyxy"], label_path)
            # На _crops — контур прямоугольного выреза чужой этикетки (magenta)
            if foreign_mask and foreign_mask.get("cuts_n"):
                try:
                    from PIL import ImageDraw as _ID

                    if crops_path.is_file():
                        crops_img, _ = open_image_rgb(crops_path)
                        draw_c = _ID.Draw(crops_img)
                        for cut in foreign_mask.get("cuts") or []:
                            rect = cut.get("rect_xyxy") or []
                            if len(rect) == 4:
                                x1, y1, x2, y2 = [float(v) for v in rect]
                                draw_c.rectangle(
                                    [x1, y1, x2, y2],
                                    outline=(255, 0, 255),
                                    width=3,
                                )
                        crops_img.save(crops_path)
                        status["crops_filename"] = crops_path.name
                except Exception as _mask_draw_exc:  # noqa: BLE001
                    _plog(
                        "WARN foreign_mask draw failed",
                        error=str(_mask_draw_exc)[:160],
                    )
            status["steps"]["label_crop"] = {
                "ok": True,
                "path": str(label_path),
                "filename": label_path.name,
                "box": [round(v, 2) for v in selected_box["xyxy"]],
                "source": label_box_source,
                "select_reason": selected_box.get("select_reason"),
                "from_image": pipeline_path.name,
                "foreign_mask": foreign_mask,
            }
            status["label_box_source"] = label_box_source
        else:
            img, _ = open_image_rgb(pipeline_path)
            w, h = img.size
            save_rgb_jpeg(img, label_path)
            # На _crops — рамка всего кадра, чтобы было видно fallback
            m = 5
            draw_boxes(
                pipeline_path,
                [
                    {
                        "xyxy": [
                            m,
                            m,
                            max(m + 1, w - 1 - m),
                            max(m + 1, h - 1 - m),
                        ],
                        "area": float(w * h),
                    }
                ],
                crops_path,
                thickness=5,
            )
            status["crops_filename"] = crops_path.name
            status["steps"]["label_crop"] = {
                "ok": True,
                "path": str(label_path),
                "filename": label_path.name,
                "box": [0, 0, w, h],
                "source": "full_image",
                "fallback": True,
                "reason": "no_label_box",
                "from_image": pipeline_path.name,
            }
        status["label_filename"] = label_path.name
        status["label_box_source"] = str(
            (status["steps"].get("label_crop") or {}).get("source")
            or label_box_source
        )
        # HF SigLIP2: JPEG short-side 384 — часть шага normalize (payload меньше)
        hf_short = int(getattr(app_config, "HF_EMBED_SHORT_SIDE", 384) or 384)
        label_hf_path = label_path.with_name(f"{label_path.stem}_hf.jpg")
        t_hf_norm = time.perf_counter()
        try:
            hf_meta = save_short_side_jpeg(
                label_path,
                label_hf_path,
                short_side=hf_short,
                quality=NORMALIZE_JPEG_QUALITY,
            )
            hf_meta["ms"] = round((time.perf_counter() - t_hf_norm) * 1000, 1)
            status["steps"]["normalize"] = {
                **(status["steps"].get("normalize") or {}),
                "hf_embed": hf_meta,
            }
            status["label_hf_filename"] = label_hf_path.name
            # Время prep включаем в полоску normalize
            prev_n = float((status.get("timings_ms") or {}).get("normalize") or 0)
            extra = float(hf_meta.get("ms") or 0)
            if extra > 0:
                status["timings_ms"]["normalize"] = round(prev_n + extra, 1)
            _plog(
                "normalize hf_embed",
                short_side=hf_short,
                size=hf_meta.get("size"),
                bytes=hf_meta.get("bytes"),
                resized=hf_meta.get("resized"),
                ms=hf_meta.get("ms"),
            )
        except Exception as hf_prep_exc:  # noqa: BLE001
            label_hf_path = label_path
            status["steps"]["normalize"] = {
                **(status["steps"].get("normalize") or {}),
                "hf_embed": {
                    "ok": False,
                    "error": str(hf_prep_exc)[:300],
                    "short_side": hf_short,
                    "fallback": "full_label",
                },
            }
            _plog("WARN normalize hf_embed failed", error=str(hf_prep_exc)[:200])
    except Exception as exc:  # noqa: BLE001
        status["steps"]["label_crop"] = {"ok": False, "error": str(exc)}
        status["ok"] = False
        status["error"] = f"crop label: {exc}"
        _plog("FAIL label_crop failed", error=str(exc))
        _time_set("label_crop", t0)
        _time_total()
        _save_status(db, row, status, commit=True)
        _shutdown_hf_start_pool(force=True)
        return status
    _plog(
        "label_crop",
        source=(status["steps"]["label_crop"] or {}).get("source"),
        fallback=(status["steps"]["label_crop"] or {}).get("fallback"),
    )
    _time_set("label_crop", t0)
    _save_status(db, row, status)

    # 3+4) Parallel: OCR/Gemini/OpenAI сразу; GPU embed ждёт HF start внутри job
    emb_step: dict[str, Any] = {}
    t_parallel = time.perf_counter()
    _plog(
        ">> parallel branch",
        embed=embedding_device,
        dinov3=use_dinov3,
        gemini_ocr="reuse" if gemini_label_detect_ran else "run",
        openai_ocr="reuse" if openai_label_detect_ran else "run",
        deepseek_ocr="run" if use_deepseek_ocr else "off",
        qwen_ocr="run" if use_qwen_ocr else "off",
        yandex_ocr="run" if use_yandex_ocr else "off",
        google_vision_ocr="run" if use_google_vision_ocr else "off",
    )

    def _job_siglip() -> dict[str, Any]:
        """GPU → remote CPU → in-process local; stages for timeline bars."""
        stages: list[dict[str, Any]] = []
        gpu_probe_sec = min(
            8.0,
            max(2.0, float(hf_start_timeout_sec)),
        )
        gpu_exc: Exception | None = None
        paused = False
        network = False

        if want_gpu_embed:
            t_gpu = time.perf_counter()
            try:
                # HF: маленький JPEG (short side 384) + Session keep-alive
                vec, meta = embed_siglip2(
                    label_hf_path,
                    force_cpu=False,
                    device="gpu",
                    timeout=gpu_probe_sec,
                )
                ms = round((time.perf_counter() - t_gpu) * 1000, 1)
                stages.append(
                    {
                        "key": "embed_siglip2_gpu",
                        "t_start": t_gpu,
                        "ms": ms,
                        "ok": True,
                    }
                )
                device = str((meta or {}).get("provider") or "gpu")
                if device not in ("cpu", "gpu"):
                    device = "gpu"
                return {
                    "ok": True,
                    "vec": vec,
                    "meta": {**(meta or {}), "device": device, "provider": device},
                    "ms": ms,
                    "t_start": t_gpu,
                    "device": device,
                    "stages": stages,
                }
            except Exception as exc:  # noqa: BLE001
                gpu_exc = exc
                ms = round((time.perf_counter() - t_gpu) * 1000, 1)
                stages.append(
                    {
                        "key": "embed_siglip2_gpu",
                        "t_start": t_gpu,
                        "ms": ms,
                        "ok": False,
                        "error": str(exc)[:300],
                    }
                )
                paused = is_hf_endpoint_paused_error(exc)
                network = is_hf_endpoint_network_error(exc)
                # Pause/400 → сразу resume в daemon-потоке, НЕ ждём;
                # дальше сразу remote CPU (и при нужде local).
                if paused or network:
                    _kick_hf_start("siglip2", t_kick=time.perf_counter())
                    _plog(
                        f"HF siglip2 {'paused' if paused else 'network'} "
                        ">> kick resume(bg) + CPU",
                        error=str(exc)[:220],
                    )
                else:
                    _plog(
                        "HF siglip2 failed >> CPU∥local",
                        error=str(exc)[:200],
                    )

        # Remote CPU (SIGLIP2_ENDPOINT)
        t_cpu = time.perf_counter()
        try:
            vec, meta = embed_siglip2_remote_cpu(label_path)
            ms = round((time.perf_counter() - t_cpu) * 1000, 1)
            stages.append(
                {
                    "key": "embed_siglip2_cpu",
                    "t_start": t_cpu,
                    "ms": ms,
                    "ok": True,
                }
            )
            meta_out = {
                **(meta or {}),
                "device": "cpu",
                "provider": "cpu",
            }
            if gpu_exc is not None:
                meta_out.update(
                    {
                        "fallback_from_gpu": True,
                        "requested_device": "gpu",
                        "gpu_error": str(gpu_exc)[:400],
                        "hf_paused": paused,
                        "hf_network_error": network,
                        "hf_start_kicked": bool(paused or network),
                        "gpu_probe_timeout_sec": gpu_probe_sec,
                    }
                )
            return {
                "ok": True,
                "vec": vec,
                "meta": meta_out,
                "ms": ms,
                "t_start": t_cpu,
                "device": "cpu",
                "stages": stages,
            }
        except Exception as cpu_exc:  # noqa: BLE001
            ms = round((time.perf_counter() - t_cpu) * 1000, 1)
            stages.append(
                {
                    "key": "embed_siglip2_cpu",
                    "t_start": t_cpu,
                    "ms": ms,
                    "ok": False,
                    "error": str(cpu_exc)[:300],
                }
            )
            _plog(
                "SigLIP2 remote CPU failed >> local",
                error=str(cpu_exc)[:200],
            )

            # In-process local (not Docker)
            if not siglip_local_enabled():
                err_parts = []
                if gpu_exc is not None:
                    err_parts.append(f"GPU ({str(gpu_exc)[:160]})")
                err_parts.append(f"CPU ({str(cpu_exc)[:180]})")
                err_parts.append("local disabled")
                return {
                    "ok": False,
                    "error": "; ".join(err_parts),
                    "ms": ms,
                    "t_start": t_cpu,
                    "device": "cpu",
                    "stages": stages,
                }

            t_local = time.perf_counter()
            try:
                vec, meta = embed_siglip2_local(label_path)
                ms_l = round((time.perf_counter() - t_local) * 1000, 1)
                stages.append(
                    {
                        "key": "embed_siglip2_local",
                        "t_start": t_local,
                        "ms": ms_l,
                        "ok": True,
                    }
                )
                meta_out = {
                    **(meta or {}),
                    "device": "local",
                    "provider": "local",
                    "fallback_from_remote_cpu": True,
                    "remote_cpu_error": str(cpu_exc)[:400],
                }
                if gpu_exc is not None:
                    meta_out.update(
                        {
                            "fallback_from_gpu": True,
                            "requested_device": "gpu",
                            "gpu_error": str(gpu_exc)[:400],
                            "hf_paused": paused,
                            "hf_network_error": network,
                            "hf_start_kicked": bool(paused or network),
                        }
                    )
                return {
                    "ok": True,
                    "vec": vec,
                    "meta": meta_out,
                    "ms": ms_l,
                    "t_start": t_local,
                    "device": "local",
                    "stages": stages,
                }
            except Exception as local_exc:  # noqa: BLE001
                ms_l = round((time.perf_counter() - t_local) * 1000, 1)
                stages.append(
                    {
                        "key": "embed_siglip2_local",
                        "t_start": t_local,
                        "ms": ms_l,
                        "ok": False,
                        "error": str(local_exc)[:300],
                    }
                )
                err_parts = []
                if gpu_exc is not None:
                    err_parts.append(f"GPU ({str(gpu_exc)[:140]})")
                err_parts.append(f"CPU ({str(cpu_exc)[:160]})")
                err_parts.append(f"local ({str(local_exc)[:180]})")
                return {
                    "ok": False,
                    "error": "; ".join(err_parts),
                    "ms": ms_l,
                    "t_start": t_local,
                    "device": "local",
                    "stages": stages,
                }

    def _job_dinov3() -> dict[str, Any]:
        if not use_dinov3:
            return {
                "ok": False,
                "skipped": True,
                "reason": "disabled_in_settings",
                "ms": 0,
            }
        if want_gpu_embed:
            return _embed_gpu_then_cpu(role="dinov3", embed_fn=embed_dinov3)
        t_embed = time.perf_counter()
        try:
            vec, meta = embed_dinov3(
                label_path, force_cpu=False, device="cpu"
            )
            return {
                "ok": True,
                "vec": vec,
                "meta": {**(meta or {}), "device": "cpu", "provider": "cpu"},
                "ms": round((time.perf_counter() - t_embed) * 1000, 1),
                "t_start": t_embed,
                "device": "cpu",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": str(exc),
                "ms": round((time.perf_counter() - t_embed) * 1000, 1),
                "t_start": t_embed,
                "device": "cpu",
            }

    def _reuse_llm_ocr_from_detect(
        *,
        engine: str,
        engine_name: str,
        variant: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Reuse OCR from label+OCR; never call the LLM again."""
        if variant is not None:
            out = {
                **variant,
                "skipped": False,
                "from_label_detect": True,
                # Время уже в *_label_detect — не рисовать вторую полоску
                "ms": 0,
                "ocr_ms": 0,
            }
            return out
        return {
            "id": engine,
            "engine": engine,
            "engine_name": engine_name,
            "ok": False,
            "skipped": True,
            "from_label_detect": True,
            "reason": "label_detect_already_ran",
            "ms": 0,
            "ocr_ms": 0,
        }

    # Не ставим в пул OCR-only, если label+OCR уже был или OCR выключен в настройках
    gemini: dict[str, Any] | None
    openai_ocr: dict[str, Any] | None
    deepseek_ocr: dict[str, Any] | None
    qwen_ocr: dict[str, Any] | None
    yandex_ocr: dict[str, Any] | None
    google_vision_ocr: dict[str, Any] | None
    if gemini_label_detect_ran:
        gemini = _reuse_llm_ocr_from_detect(
            engine="gemini",
            engine_name="Google Gemini",
            variant=gemini_from_detect,
        )
    elif not use_gemini_ocr:
        gemini = {
            "id": "gemini",
            "engine": "gemini",
            "engine_name": "Google Gemini",
            "ok": False,
            "skipped": True,
            "reason": "disabled_in_settings",
            "ms": 0,
            "ocr_ms": 0,
        }
    else:
        gemini = None
    if openai_label_detect_ran:
        openai_ocr = _reuse_llm_ocr_from_detect(
            engine="openai",
            engine_name="OpenAI",
            variant=openai_from_detect,
        )
    elif not use_openai_ocr:
        openai_ocr = {
            "id": "openai",
            "engine": "openai",
            "engine_name": "OpenAI",
            "ok": False,
            "skipped": True,
            "reason": "disabled_in_settings",
            "ms": 0,
            "ocr_ms": 0,
        }
    else:
        openai_ocr = None
    if not use_deepseek_ocr:
        deepseek_ocr = {
            "id": "deepseek",
            "engine": "deepseek",
            "engine_name": "DeepSeek",
            "ok": False,
            "skipped": True,
            "reason": "disabled_in_settings",
            "ms": 0,
            "ocr_ms": 0,
        }
    else:
        deepseek_ocr = None
    if not use_qwen_ocr:
        qwen_ocr = {
            "id": "qwen",
            "engine": "qwen",
            "engine_name": "Qwen2.5-VL (HF)",
            "ok": False,
            "skipped": True,
            "reason": "disabled_in_settings",
            "ms": 0,
            "ocr_ms": 0,
        }
    else:
        qwen_ocr = None
    if not use_yandex_ocr:
        yandex_ocr = {
            "id": "yandex",
            "engine": "yandex",
            "engine_name": "Yandex OCR",
            "ok": False,
            "skipped": True,
            "reason": "disabled_in_settings",
            "ms": 0,
            "ocr_ms": 0,
        }
    else:
        yandex_ocr = None
    if not use_google_vision_ocr:
        google_vision_ocr = {
            "id": "google_vision",
            "engine": "google_vision",
            "engine_name": "Google Vision OCR",
            "ok": False,
            "skipped": True,
            "reason": "disabled_in_settings",
            "ms": 0,
            "ocr_ms": 0,
        }
    else:
        google_vision_ocr = None

    def _job_ocr() -> dict[str, Any]:
        try:
            return run_ocr_variants(label_path)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _job_gemini() -> dict[str, Any]:
        return run_gemini_ocr(label_path)

    def _job_openai() -> dict[str, Any]:
        return run_openai_ocr(label_path)

    def _job_deepseek() -> dict[str, Any]:
        return run_deepseek_ocr(label_path)

    def _job_yandex() -> dict[str, Any]:
        return run_yandex_ocr(label_path)

    def _job_google_vision() -> dict[str, Any]:
        return run_google_vision_ocr(label_path)

    def _job_qwen() -> dict[str, Any]:
        # Без pre-start: пробуем OCR; при pause — kick start (на потом) и skip
        t_ocr = time.perf_counter()
        out = run_qwen_ocr(label_path)
        out["t_start"] = t_ocr
        if out.get("ok") or out.get("skipped"):
            return out
        err = str(out.get("error") or "")
        if need_hf_qwen and is_hf_endpoint_paused_error(err):
            _kick_hf_start("qwen_ocr")
            out["skipped"] = True
            out["reason"] = "hf_endpoint_paused_start_kicked"
            out["note"] = (
                "HF Qwen на паузе — старт в фоне для следующего скана; "
                "OCR в этом прогоне пропущен"
            )
            _plog("HF qwen_ocr paused >> kick start, skip OCR this run")
        return out

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=11)
    try:
        # OCR/LLM сразу; GPU embed — прямой вызов (без ожидания HF start)
        fut_ocr = pool.submit(_job_ocr)
        fut_gemini = (
            pool.submit(_job_gemini) if gemini is None else None
        )
        fut_openai = (
            pool.submit(_job_openai) if openai_ocr is None else None
        )
        fut_deepseek = (
            pool.submit(_job_deepseek) if deepseek_ocr is None else None
        )
        fut_yandex = (
            pool.submit(_job_yandex) if yandex_ocr is None else None
        )
        # Live Google Vision: если reuse_previous_searches — ждём cosine;
        # при hit с GV-текстом прошлого поиска live API не вызываем.
        fut_google_vision = None
        if google_vision_ocr is None and not reuse_previous_searches:
            fut_google_vision = pool.submit(_job_google_vision)
        fut_qwen = pool.submit(_job_qwen) if qwen_ocr is None else None
        fut_sig = pool.submit(_job_siglip)
        fut_dino = pool.submit(_job_dinov3)

        # Wait embeddings first (needed for candidates / geometry);
        # OCR + Gemini + OpenAI keep running in the background.
        res_sig = fut_sig.result()
        res_dino = fut_dino.result()
        # На случай если embed skipped — всё равно зафиксировать HF steps
        _flush_hf_start_status()
        if embedding_device == "gpu":
            _save_status(db, row, status)

        # Embed bars: SigLIP2 stages (GPU / remote CPU / local); DINOv3 as before
        sig_stages = res_sig.get("stages") or []
        if sig_stages:
            for st in sig_stages:
                if not isinstance(st, dict):
                    continue
                key = str(st.get("key") or "").strip()
                ms = float(st.get("ms") or 0)
                if not key or ms <= 0:
                    continue
                t_emb = st.get("t_start") or t_parallel
                _time_set(key, t_emb, duration_ms=ms)
        elif res_sig.get("ms") and float(res_sig.get("ms") or 0) > 0:
            t_emb = res_sig.get("t_start") or t_parallel
            _time_set("embed_siglip2", t_emb, duration_ms=res_sig.get("ms"))
        if (
            not res_dino.get("skipped")
            and res_dino.get("ms")
            and float(res_dino.get("ms") or 0) > 0
        ):
            t_emb = res_dino.get("t_start") or t_parallel
            _time_set("embed_dinov3", t_emb, duration_ms=res_dino.get("ms"))

        if res_sig.get("ok") and res_sig.get("vec") is not None:
            try:
                emb_id_s = _insert_embedding(
                    db,
                    search_photos_id=photo_id,
                    embedding_type="siglip2",
                    embedding=res_sig["vec"],
                )
                meta_s = dict(res_sig.get("meta") or {})
                device_s = str(
                    res_sig.get("device")
                    or meta_s.get("device")
                    or meta_s.get("provider")
                    or ""
                ).lower()
                if device_s not in ("cpu", "gpu", "local"):
                    device_s = "cpu"
                emb_step["siglip2"] = {
                    "ok": True,
                    "id": emb_id_s,
                    "dim": len(res_sig["vec"]),
                    **meta_s,
                    "device": device_s,
                    "provider": meta_s.get("provider") or device_s,
                }
            except Exception as exc:  # noqa: BLE001
                emb_step["siglip2"] = {
                    "ok": False,
                    "error": str(exc),
                    "device": res_sig.get("device"),
                }
        else:
            emb_step["siglip2"] = {
                k: v
                for k, v in res_sig.items()
                if k not in {"vec", "meta"}
            }
            if "device" not in emb_step["siglip2"] and res_sig.get("device"):
                emb_step["siglip2"]["device"] = res_sig.get("device")

        if res_dino.get("skipped"):
            emb_step["dinov3"] = {
                "ok": False,
                "skipped": True,
                "reason": res_dino.get("reason") or "disabled_in_settings",
            }
        elif res_dino.get("ok") and res_dino.get("vec") is not None:
            try:
                emb_id_d = _insert_embedding(
                    db,
                    search_photos_id=photo_id,
                    embedding_type="dinov3",
                    embedding=res_dino["vec"],
                )
                meta_d = dict(res_dino.get("meta") or {})
                device_d = str(
                    res_dino.get("device")
                    or meta_d.get("device")
                    or meta_d.get("provider")
                    or ""
                ).lower()
                if device_d not in ("cpu", "gpu"):
                    device_d = "cpu"
                emb_step["dinov3"] = {
                    "ok": True,
                    "id": emb_id_d,
                    "dim": len(res_dino["vec"]),
                    **meta_d,
                    "device": device_d,
                    "provider": meta_d.get("provider") or device_d,
                }
            except Exception as exc:  # noqa: BLE001
                emb_step["dinov3"] = {
                    "ok": False,
                    "error": str(exc),
                    "device": res_dino.get("device"),
                }
        else:
            emb_step["dinov3"] = {
                k: v
                for k, v in res_dino.items()
                if k not in {"vec", "meta"}
            }
            if "device" not in emb_step["dinov3"] and res_dino.get("device"):
                emb_step["dinov3"]["device"] = res_dino.get("device")

        status["steps"]["embeddings"] = emb_step
        # Явные признаки фактического endpoint (не из настроек)
        def _emb_device(ch: str) -> str | None:
            st = emb_step.get(ch) or {}
            d = str(st.get("device") or st.get("provider") or "").lower()
            return d if d in ("cpu", "gpu", "local") else None

        status["embed_siglip2_device"] = _emb_device("siglip2")
        status["embed_dinov3_device"] = _emb_device("dinov3")
        _plog(
            "embeddings",
            siglip=emb_step.get("siglip2", {}).get("ok"),
            dinov3=(
                "skip"
                if emb_step.get("dinov3", {}).get("skipped")
                else emb_step.get("dinov3", {}).get("ok")
            ),
            siglip_ms=res_sig.get("ms"),
            dino_ms=None if res_dino.get("skipped") else res_dino.get("ms"),
        )
        _save_status(db, row, status)

        # 3b) cosine top-N — overlaps remaining OCR/Gemini work
        _plog(
            ">> candidates (cosine)",
            reuse=reuse_previous_searches,
            top_n=candidates_top_n,
        )
        t0 = time.perf_counter()
        candidates = _run_cosine_search(
            db,
            search_photos_id=photo_id,
            emb_step=emb_step,
            reuse_previous_searches=reuse_previous_searches,
            candidates_top_n=candidates_top_n,
        )
        reuse_step = candidates.pop("reuse_previous_search", None)
        status["steps"]["candidates"] = candidates
        status["candidates_siglip2_ids"] = candidates.get("siglip2", {}).get("ids", [])
        status["candidates_dinov3_ids"] = candidates.get("dinov3", {}).get("ids", [])
        if reuse_step:
            status["steps"]["reuse_previous_search"] = reuse_step
            status["reuse_previous_search"] = reuse_step
            # Also expose GV OCR at top-level for clients
            status["reused_google_vision_ocr"] = reuse_step.get("google_vision_ocr")
        # Deferred live GV: only if reuse did not supply OCR text.
        reused_gv_text = ""
        if isinstance(reuse_step, dict):
            reused_gv_text = (reuse_step.get("google_vision_ocr") or "").strip()
        if (
            google_vision_ocr is None
            and reuse_previous_searches
            and fut_google_vision is None
        ):
            if reused_gv_text:
                google_vision_ocr = {
                    "id": "google_vision",
                    "engine": "google_vision",
                    "engine_name": "Google Vision OCR",
                    "ok": True,
                    "text": reused_gv_text,
                    "reused": True,
                    "reused_from_search_photos_id": (
                        reuse_step or {}
                    ).get("search_photos_id"),
                    "skipped_live": True,
                    "reason": "reuse_previous_search_gv",
                }
                _plog(
                    "google_vision skipped (reuse OCR)",
                    from_sid=(reuse_step or {}).get("search_photos_id"),
                    wine_id=(reuse_step or {}).get("wine_id"),
                )
            else:
                fut_google_vision = pool.submit(_job_google_vision)
                _plog(">> google_vision live (no reuse OCR)")
        cos_persist: dict[str, float] = {}
        for branch in ("siglip2", "dinov3"):
            for it in (candidates.get(branch) or {}).get("items") or []:
                if not isinstance(it, dict) or it.get("id") is None:
                    continue
                wid = str(int(it["id"]))
                try:
                    v = float(it.get("cosine_similarity"))
                except (TypeError, ValueError):
                    continue
                prev = cos_persist.get(wid)
                if prev is None or v > prev:
                    cos_persist[wid] = round(v, 6)
        status["candidates_cosine"] = cos_persist
        _plog(
            "candidates",
            siglip_n=len(status["candidates_siglip2_ids"] or []),
            dino_n=len(status["candidates_dinov3_ids"] or []),
            reuse=bool(reuse_step),
            reuse_wine=(reuse_step or {}).get("wine_id"),
            reuse_sid=(reuse_step or {}).get("search_photos_id"),
        )
        _time_set("candidates", t0)
        _save_status(db, row, status)

        # 3b2) HSV Bhattacharyya: query label vs each top-20 catalog crop
        _plog(">> hsv")
        t0 = time.perf_counter()
        if not compute_hsv:
            hsv_step = {
                "ok": False,
                "skipped": True,
                "reason": "compute_hsv=false",
                "metric": "bhattacharyya",
                "by_id": {},
                "missing": [],
                "n": 0,
                "winner": None,
            }
        else:
            try:
                hsv_step = score_candidates_hsv(label_path, candidates, db=db)
            except Exception as exc:  # noqa: BLE001
                hsv_step = {
                    "ok": False,
                    "error": str(exc),
                    "metric": "bhattacharyya",
                    "by_id": {},
                    "missing": [],
                    "n": 0,
                    "winner": None,
                }
        status["steps"]["hsv"] = hsv_step
        # Optional soft filter: hsv > max → unsuitable (later ∪ exclusive)
        # Optionally keep if max cosine ≥ hsv_ignore_cosine_min.
        # Absolute hard reject: hsv > hsv_hard_reject_max → always unsuitable
        # (no cosine bypass). Gates require compute_hsv.
        hsv_reject_ids: list[int] = []
        hsv_cosine_kept_ids: list[int] = []
        hsv_hard_reject_ids: list[int] = []
        hsv_step["compute_enabled"] = bool(compute_hsv)
        hsv_step["filter_enabled"] = bool(use_hsv_filter)
        hsv_step["threshold"] = float(hsv_bhattacharyya_max)
        hsv_step["ignore_high_cosine"] = bool(hsv_ignore_high_cosine)
        hsv_step["ignore_cosine_min"] = float(hsv_ignore_cosine_min)
        hsv_step["hard_reject_enabled"] = bool(use_hsv_hard_reject)
        hsv_step["hard_reject_max"] = float(hsv_hard_reject_max)
        hsv_step["reject_rule"] = (
            "skipped (compute_hsv=false)"
            if not compute_hsv
            else (
                "hsv > threshold → unsuitable"
                + (
                    f"; keep if cosine ≥ {float(hsv_ignore_cosine_min):.2f}"
                    if hsv_ignore_high_cosine
                    else ""
                )
                + (
                    f"; hard reject if hsv > {float(hsv_hard_reject_max):.2f}"
                    if use_hsv_hard_reject
                    else ""
                )
            )
        )
        if hsv_step.get("ok") and (use_hsv_filter or use_hsv_hard_reject):
            by_hsv = hsv_step.get("by_id") or {}
            thr = float(hsv_bhattacharyya_max)
            cos_keep = float(hsv_ignore_cosine_min)
            hard_thr = float(hsv_hard_reject_max)
            for wid_s, dist in by_hsv.items():
                try:
                    d = float(dist)
                    wid_i = int(wid_s)
                except (TypeError, ValueError):
                    continue
                rejected = bool(use_hsv_filter) and d > thr
                kept_by_cosine = False
                if rejected and hsv_ignore_high_cosine:
                    try:
                        cos_v = float(cos_persist.get(str(wid_i)))
                    except (TypeError, ValueError):
                        cos_v = None
                    if cos_v is not None and cos_v >= cos_keep:
                        rejected = False
                        kept_by_cosine = True
                        hsv_cosine_kept_ids.append(wid_i)
                hard_hit = bool(use_hsv_hard_reject) and d > hard_thr
                if hard_hit:
                    rejected = True
                    kept_by_cosine = False
                    hsv_hard_reject_ids.append(wid_i)
                    if wid_i in hsv_cosine_kept_ids:
                        hsv_cosine_kept_ids = [
                            x for x in hsv_cosine_kept_ids if x != wid_i
                        ]
                if rejected:
                    hsv_reject_ids.append(wid_i)
                for branch in ("siglip2", "dinov3"):
                    items = (candidates.get(branch) or {}).get("items") or []
                    for it in items:
                        if isinstance(it, dict) and int(it.get("id") or 0) == wid_i:
                            it["hsv_rejected"] = rejected
                            it["hsv_suitable"] = not rejected
                            if kept_by_cosine:
                                it["hsv_kept_by_cosine"] = True
                            if hard_hit:
                                it["hsv_hard_rejected"] = True
            hsv_reject_ids = sorted(set(hsv_reject_ids))
            hsv_cosine_kept_ids = sorted(set(hsv_cosine_kept_ids))
            hsv_hard_reject_ids = sorted(set(hsv_hard_reject_ids))
        hsv_step["rejected_ids"] = hsv_reject_ids
        hsv_step["kept_by_cosine_ids"] = hsv_cosine_kept_ids
        hsv_step["hard_rejected_ids"] = hsv_hard_reject_ids
        status["hsv_rejected_ids"] = hsv_reject_ids
        _plog(
            "hsv",
            ok=hsv_step.get("ok"),
            skipped=not compute_hsv,
            n=hsv_step.get("n"),
            from_db=hsv_step.get("from_db"),
            computed=hsv_step.get("computed"),
            persisted=hsv_step.get("persisted"),
            missing=len(hsv_step.get("missing") or []),
            filter=use_hsv_filter,
            rejected=len(hsv_reject_ids),
            kept_cos=len(hsv_cosine_kept_ids),
            hard=use_hsv_hard_reject,
            hard_n=len(hsv_hard_reject_ids),
            thr=hsv_bhattacharyya_max,
            hard_thr=hsv_hard_reject_max,
            ignore_cos=hsv_ignore_high_cosine,
            cos_min=hsv_ignore_cosine_min,
        )
        _time_set("hsv", t0)
        _save_status(db, row, status)

        # 3b3) Dominant-color CIEDE2000 (ColorDelta)
        _plog(">> color_delta")
        t0 = time.perf_counter()
        if not compute_color_delta:
            color_delta_step: dict[str, Any] = {
                "ok": False,
                "skipped": True,
                "reason": "compute_color_delta=false",
                "metric": "ciede2000_dominant",
                "by_id": {},
                "missing": [],
                "n": 0,
                "winner": None,
            }
        else:
            try:
                color_delta_step = score_candidates_color_delta(
                    label_path, candidates
                )
            except Exception as exc:  # noqa: BLE001
                color_delta_step = {
                    "ok": False,
                    "error": str(exc),
                    "metric": "ciede2000_dominant",
                    "by_id": {},
                    "missing": [],
                    "n": 0,
                    "winner": None,
                }
        color_delta_step["compute_enabled"] = bool(compute_color_delta)
        status["steps"]["color_delta"] = color_delta_step
        _plog(
            "color_delta",
            ok=color_delta_step.get("ok"),
            skipped=not compute_color_delta,
            n=color_delta_step.get("n"),
        )
        _time_set("color_delta", t0)
        _save_status(db, row, status)

        # 3c) geometric verification (also overlaps OCR/Gemini if still running)
        _plog(
            ">> geometry",
            siglip2=geometry_siglip2,
            dinov3=geometry_dinov3 and use_dinov3,
        )
        t0 = time.perf_counter()
        geometry_step: dict[str, Any] = {
            "ok": True,
            "channels": {},
        }
        for channel, enabled in (
            ("siglip2", geometry_siglip2),
            ("dinov3", geometry_dinov3 and use_dinov3),
        ):
            ids = list(candidates.get(channel, {}).get("ids") or [])
            if not enabled:
                geometry_step["channels"][channel] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "disabled_in_settings",
                    "items": [],
                }
                continue
            if not ids:
                geometry_step["channels"][channel] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "no_candidates",
                    "items": [],
                }
                continue
            try:
                geo = verify_candidates_geometry(
                    db, query_label_path=label_path, candidate_ids=ids
                )
                by_id = geo.pop("by_id", {}) or {}
                geometry_step["channels"][channel] = geo
                items = candidates.get(channel, {}).get("items") or []
                if candidates.get(channel):
                    candidates[channel]["items"] = attach_geometry_to_items(
                        items, by_id, resort=False
                    )
                    candidates[channel]["ids"] = [
                        int(x["id"]) for x in candidates[channel]["items"]
                    ]
                    status[f"candidates_{channel}_ids"] = candidates[channel]["ids"]
            except Exception as exc:  # noqa: BLE001
                geometry_step["channels"][channel] = {
                    "ok": False,
                    "error": str(exc),
                    "items": [],
                }
        status["steps"]["candidates"] = candidates
        status["steps"]["geometry"] = geometry_step
        _plog(
            "geometry done",
            siglip=(geometry_step.get("channels") or {}).get("siglip2", {}).get("ok"),
            dinov3=(geometry_step.get("channels") or {}).get("dinov3", {}).get("ok"),
        )
        _time_set("geometry", t0)
        _save_status(db, row, status)

        # Collect OCR + Gemini + OpenAI + DeepSeek + Yandex + Google Vision + Qwen
        _plog(">> wait OCR / Gemini / OpenAI / DeepSeek / Yandex / GoogleVision / Qwen")
        t_ocr_wait = time.perf_counter()
        try:
            ocr = fut_ocr.result()
        except Exception as exc:  # noqa: BLE001
            ocr = {"ok": False, "error": str(exc), "variants": {}}
        if fut_gemini is not None:
            try:
                gemini = fut_gemini.result()
            except Exception as exc:  # noqa: BLE001
                gemini = {"id": "gemini", "ok": False, "error": str(exc)}
        if fut_openai is not None:
            try:
                openai_ocr = fut_openai.result()
            except Exception as exc:  # noqa: BLE001
                openai_ocr = {"id": "openai", "ok": False, "error": str(exc)}
        if fut_deepseek is not None:
            try:
                deepseek_ocr = fut_deepseek.result()
            except Exception as exc:  # noqa: BLE001
                deepseek_ocr = {"id": "deepseek", "ok": False, "error": str(exc)}
        if fut_yandex is not None:
            try:
                yandex_ocr = fut_yandex.result()
            except Exception as exc:  # noqa: BLE001
                yandex_ocr = {"id": "yandex", "ok": False, "error": str(exc)}
        if fut_google_vision is not None:
            try:
                google_vision_ocr = fut_google_vision.result()
            except Exception as exc:  # noqa: BLE001
                google_vision_ocr = {
                    "id": "google_vision",
                    "ok": False,
                    "error": str(exc),
                }
        if fut_qwen is not None:
            try:
                qwen_ocr = fut_qwen.result()
            except Exception as exc:  # noqa: BLE001
                qwen_ocr = {"id": "qwen", "ok": False, "error": str(exc)}
    finally:
        pool.shutdown(wait=True)

    def _llm_ocr_broken(res: Any) -> bool:
        """True if engine was attempted and failed (not merely disabled)."""
        if not isinstance(res, dict):
            return False
        if res.get("reason") == "disabled_in_settings":
            return False
        return not bool(res.get("ok"))

    # Vision выключен в настройках, но OpenAI и/или Gemini упали → один раз Vision
    if not use_google_vision_ocr:
        broken_from: list[str] = []
        if _llm_ocr_broken(openai_ocr):
            broken_from.append("openai")
        if _llm_ocr_broken(gemini):
            broken_from.append("gemini")
        if broken_from:
            _plog(
                ">> google_vision OCR fallback",
                because=",".join(broken_from),
            )
            t_gv = time.perf_counter()
            try:
                google_vision_ocr = run_google_vision_ocr(label_path, force=True)
            except Exception as exc:  # noqa: BLE001
                google_vision_ocr = {
                    "id": "google_vision",
                    "engine": "google_vision",
                    "engine_name": "Google Vision OCR",
                    "ok": False,
                    "error": str(exc),
                }
            if isinstance(google_vision_ocr, dict):
                google_vision_ocr["fallback"] = True
                google_vision_ocr["fallback_from"] = broken_from
                google_vision_ocr["fallback_reason"] = (
                    "openai_or_gemini_failed_vision_disabled_in_settings"
                )
                google_vision_ocr["t_start"] = t_gv

    if not isinstance(ocr, dict):
        ocr = {"ok": False, "error": "bad ocr result", "variants": {}}
    _plog(
        "OCR collected",
        local=bool(ocr.get("ok")),
        gemini=(
            "reuse"
            if isinstance(gemini, dict) and gemini.get("from_label_detect")
            else (isinstance(gemini, dict) and gemini.get("ok"))
        ),
        openai=(
            "reuse"
            if isinstance(openai_ocr, dict) and openai_ocr.get("from_label_detect")
            else (isinstance(openai_ocr, dict) and openai_ocr.get("ok"))
        ),
        deepseek=(isinstance(deepseek_ocr, dict) and deepseek_ocr.get("ok")),
        yandex=(isinstance(yandex_ocr, dict) and yandex_ocr.get("ok")),
        google_vision=(
            isinstance(google_vision_ocr, dict) and google_vision_ocr.get("ok")
        ),
        google_vision_fallback=(
            isinstance(google_vision_ocr, dict) and google_vision_ocr.get("fallback")
        ),
        qwen=(
            isinstance(qwen_ocr, dict)
            and (qwen_ocr.get("ok") or qwen_ocr.get("skipped"))
        ),
    )
    variants = dict(ocr.get("variants") or {})
    if isinstance(gemini, dict):
        variants["gemini"] = gemini
        # Prefer Gemini full_text as primary when available
        gtext = (gemini.get("text") or "").strip()
        if gtext:
            ocr["text"] = gtext
            ocr["text_norm"] = normalize_ocr_text(gtext)
            agg = ocr.get("text_aggregated") or ""
            if gtext not in agg:
                ocr["text_aggregated"] = (gtext + ("\n" + agg if agg else "")).strip()
        ocr["ok"] = bool(ocr.get("ok")) or bool(gemini.get("ok"))
    if isinstance(openai_ocr, dict):
        variants["openai"] = openai_ocr
        otext = (openai_ocr.get("text") or "").strip()
        if otext:
            agg = ocr.get("text_aggregated") or ""
            if otext not in agg:
                ocr["text_aggregated"] = (otext + ("\n" + agg if agg else "")).strip()
            if not (ocr.get("text") or "").strip():
                ocr["text"] = otext
                ocr["text_norm"] = normalize_ocr_text(otext)
        ocr["ok"] = bool(ocr.get("ok")) or bool(openai_ocr.get("ok"))
    if isinstance(deepseek_ocr, dict):
        variants["deepseek"] = deepseek_ocr
        dtext = (deepseek_ocr.get("text") or "").strip()
        if dtext:
            agg = ocr.get("text_aggregated") or ""
            if dtext not in agg:
                ocr["text_aggregated"] = (dtext + ("\n" + agg if agg else "")).strip()
            if not (ocr.get("text") or "").strip():
                ocr["text"] = dtext
                ocr["text_norm"] = normalize_ocr_text(dtext)
        ocr["ok"] = bool(ocr.get("ok")) or bool(deepseek_ocr.get("ok"))
    if isinstance(yandex_ocr, dict):
        variants["yandex"] = yandex_ocr
        ytext = (yandex_ocr.get("text") or "").strip()
        if ytext:
            agg = ocr.get("text_aggregated") or ""
            if ytext not in agg:
                ocr["text_aggregated"] = (ytext + ("\n" + agg if agg else "")).strip()
            if not (ocr.get("text") or "").strip():
                ocr["text"] = ytext
                ocr["text_norm"] = normalize_ocr_text(ytext)
        ocr["ok"] = bool(ocr.get("ok")) or bool(yandex_ocr.get("ok"))
    if isinstance(google_vision_ocr, dict):
        variants["google_vision"] = google_vision_ocr
        gvtext = (google_vision_ocr.get("text") or "").strip()
        if gvtext:
            agg = ocr.get("text_aggregated") or ""
            if gvtext not in agg:
                ocr["text_aggregated"] = (
                    gvtext + ("\n" + agg if agg else "")
                ).strip()
            if not (ocr.get("text") or "").strip():
                ocr["text"] = gvtext
                ocr["text_norm"] = normalize_ocr_text(gvtext)
        ocr["ok"] = bool(ocr.get("ok")) or bool(google_vision_ocr.get("ok"))

    # OCR from a reused previous search (siglip2 nearest ≥0.5).
    reuse_step_now = (status.get("steps") or {}).get("reuse_previous_search")
    if isinstance(reuse_step_now, dict):
        reused_gv = (reuse_step_now.get("google_vision_ocr") or "").strip()
        if reused_gv:
            reused_pack = {
                "id": "google_vision_reused",
                "engine": "google_vision_reused",
                "ok": True,
                "text": reused_gv,
                "from_previous_search": True,
                "search_photos_id": reuse_step_now.get("search_photos_id"),
                "wine_id": reuse_step_now.get("wine_id"),
                "cosine": reuse_step_now.get("cosine"),
            }
            variants["google_vision_reused"] = reused_pack
            status["steps"]["ocr_google_vision_reused"] = reused_pack
            agg = ocr.get("text_aggregated") or ""
            if reused_gv not in agg:
                ocr["text_aggregated"] = (
                    reused_gv + ("\n" + agg if agg else "")
                ).strip()
            if not (ocr.get("text") or "").strip():
                ocr["text"] = reused_gv
                ocr["text_norm"] = normalize_ocr_text(reused_gv)
            ocr["ok"] = True
            # Prefer exposing reused GV as primary google_vision when live GV empty.
            live_gv = ""
            if isinstance(google_vision_ocr, dict):
                live_gv = (google_vision_ocr.get("text") or "").strip()
            if not live_gv:
                google_vision_ocr = {
                    "id": "google_vision",
                    "engine": "google_vision",
                    "ok": True,
                    "text": reused_gv,
                    "reused_from_search_photos_id": reuse_step_now.get(
                        "search_photos_id"
                    ),
                    "reused": True,
                }
                variants["google_vision"] = google_vision_ocr

    if isinstance(qwen_ocr, dict):
        variants["qwen"] = qwen_ocr
        qtext = (qwen_ocr.get("text") or "").strip()
        if qtext:
            agg = ocr.get("text_aggregated") or ""
            if qtext not in agg:
                ocr["text_aggregated"] = (qtext + ("\n" + agg if agg else "")).strip()
            if not (ocr.get("text") or "").strip():
                ocr["text"] = qtext
                ocr["text_norm"] = normalize_ocr_text(qtext)
        ocr["ok"] = bool(ocr.get("ok")) or bool(qwen_ocr.get("ok"))
    ocr["variants"] = variants
    ocr["gemini"] = gemini if isinstance(gemini, dict) else None
    ocr["openai"] = openai_ocr if isinstance(openai_ocr, dict) else None
    ocr["deepseek"] = deepseek_ocr if isinstance(deepseek_ocr, dict) else None
    ocr["yandex"] = yandex_ocr if isinstance(yandex_ocr, dict) else None
    ocr["google_vision"] = (
        google_vision_ocr if isinstance(google_vision_ocr, dict) else None
    )
    ocr["qwen"] = qwen_ocr if isinstance(qwen_ocr, dict) else None
    status["steps"]["ocr"] = ocr
    status["steps"]["ocr_gemini"] = gemini if isinstance(gemini, dict) else None
    status["steps"]["ocr_openai"] = openai_ocr if isinstance(openai_ocr, dict) else None
    status["steps"]["ocr_deepseek"] = (
        deepseek_ocr if isinstance(deepseek_ocr, dict) else None
    )
    status["steps"]["ocr_yandex"] = (
        yandex_ocr if isinstance(yandex_ocr, dict) else None
    )
    status["steps"]["ocr_google_vision"] = (
        google_vision_ocr if isinstance(google_vision_ocr, dict) else None
    )
    status["steps"]["ocr_qwen"] = qwen_ocr if isinstance(qwen_ocr, dict) else None

    for key, info in (ocr.get("preprocess") or {}).items():
        if not isinstance(info, dict):
            continue
        prep_ms = info.get("preprocess_ms")
        if not prep_ms or float(prep_ms) <= 0:
            continue
        # started_ms relative to OCR job; absolute = parallel start + relative
        rel = info.get("started_ms")
        if rel is not None:
            t_abs = t_parallel + float(rel) / 1000.0
            _time_set(f"ocr_{key}_preprocess", t_abs, duration_ms=prep_ms)
        else:
            _time_set(f"ocr_{key}_preprocess", t_parallel, duration_ms=prep_ms)
    for key, v in variants.items():
        if not isinstance(v, dict) or v.get("alias_of") or v.get("skipped"):
            continue
        # OCR уже учтён в *_label_detect — не дублировать полоску в Gantt
        if v.get("from_label_detect"):
            continue
        ocr_ms = v.get("ocr_ms") or v.get("ms")
        if not ocr_ms or float(ocr_ms) <= 0:
            continue
        rel = v.get("started_ms")
        if rel is not None:
            t_abs = t_parallel + float(rel) / 1000.0
            _time_set(f"ocr_{key}", t_abs, duration_ms=ocr_ms)
        else:
            _time_set(f"ocr_{key}", t_parallel, duration_ms=ocr_ms)

    deskew_ms = ocr.get("deskew_ms")
    if deskew_ms and float(deskew_ms) > 0:
        deskew_rel = ocr.get("deskew_started_ms")
        if deskew_rel is not None:
            _time_set(
                "ocr_deskew",
                t_parallel + float(deskew_rel) / 1000.0,
                duration_ms=deskew_ms,
            )
        else:
            _time_set("ocr_deskew", t_parallel, duration_ms=deskew_ms)

    if ocr.get("ms") and float(ocr.get("ms") or 0) > 0 and not ocr.get("skipped"):
        _time_set("ocr", t_parallel, duration_ms=ocr.get("ms"))

    if (
        isinstance(gemini, dict)
        and not gemini.get("skipped")
        and not gemini.get("from_label_detect")
        and gemini.get("ms")
        and float(gemini.get("ms") or 0) > 0
    ):
        _time_set("ocr_gemini", t_parallel, duration_ms=gemini.get("ms"))
    if (
        isinstance(openai_ocr, dict)
        and not openai_ocr.get("skipped")
        and not openai_ocr.get("from_label_detect")
        and openai_ocr.get("ms")
        and float(openai_ocr.get("ms") or 0) > 0
    ):
        _time_set("ocr_openai", t_parallel, duration_ms=openai_ocr.get("ms"))
    if (
        isinstance(deepseek_ocr, dict)
        and not deepseek_ocr.get("skipped")
        and deepseek_ocr.get("ms")
        and float(deepseek_ocr.get("ms") or 0) > 0
    ):
        _time_set("ocr_deepseek", t_parallel, duration_ms=deepseek_ocr.get("ms"))
    if (
        isinstance(yandex_ocr, dict)
        and not yandex_ocr.get("skipped")
        and yandex_ocr.get("ms")
        and float(yandex_ocr.get("ms") or 0) > 0
    ):
        _time_set("ocr_yandex", t_parallel, duration_ms=yandex_ocr.get("ms"))
    if (
        isinstance(google_vision_ocr, dict)
        and not google_vision_ocr.get("skipped")
        and google_vision_ocr.get("ms")
        and float(google_vision_ocr.get("ms") or 0) > 0
    ):
        t_gv = google_vision_ocr.get("t_start") or t_parallel
        _time_set(
            "ocr_google_vision",
            t_gv,
            duration_ms=google_vision_ocr.get("ms"),
        )
    if (
        isinstance(qwen_ocr, dict)
        and not qwen_ocr.get("skipped")
        and qwen_ocr.get("ms")
        and float(qwen_ocr.get("ms") or 0) > 0
    ):
        t_q = qwen_ocr.get("t_start") or t_parallel
        _time_set("ocr_qwen", t_q, duration_ms=qwen_ocr.get("ms"))

    wait_ms = (time.perf_counter() - t_ocr_wait) * 1000
    if wait_ms > 0.5:
        _time_set("ocr_wait_after_geometry", t_ocr_wait, duration_ms=wait_ms)
    _time_set("parallel_branch", t_parallel)

    # 4a) exclusive_lexicon ∥ XGBoost ∥ CrossEncoder (после top-20 + OCR)
    pipe_settings = get_settings()
    text_methods = set(pipe_settings.get("text_match_methods") or [])
    use_xgb = "xgb" in text_methods
    use_crenc_local = "crenc" in text_methods
    use_crenc_srv = "crenc_srv" in text_methods
    use_crenc = use_crenc_local or use_crenc_srv
    crenc_backend = "server" if use_crenc_srv else "local"
    use_openai_txt = "openai_txt" in text_methods
    use_fin1 = "fin1" in text_methods
    use_fin2 = "fin2" in text_methods
    final_method = str(pipe_settings.get("final_score_method") or "fin1")
    _plog(
        f">> xgb_match ∥ crenc ({crenc_backend})"
        " → exclusive_lexicon (all top-N candidates)"
        + (" → openai_txt" if use_openai_txt else "")
    )
    ocr_step = status.get("steps", {}).get("ocr") or {}
    exclusive_query = (
        str(ocr_step.get("text") or "").strip()
        or str(ocr_step.get("text_aggregated") or "").strip()
        or str(ocr_step.get("text_norm") or "").strip()
    )
    final_ocr_engine, xgb_ocr_text = resolve_final_ocr(
        status, prefer=final_ocr_pref
    )
    if not str(xgb_ocr_text or "").strip():
        xgb_ocr_text = exclusive_query
    ocr_texts_by_engine = list_ocr_texts_by_engine(status)
    status["final_ocr"] = final_ocr_pref
    status["final_ocr_resolved"] = final_ocr_engine
    status["ocr_engines_available"] = sorted(ocr_texts_by_engine.keys())
    ocr_text_present = bool(str(xgb_ocr_text or "").strip())
    status["ocr_text_present"] = ocr_text_present
    empty_ocr_cos_thr = _clamp099(
        pipe_settings.get("empty_ocr_cosine_threshold"), 0.86
    )
    xgb_dead_max = _clamp099(pipe_settings.get("xgb_dead_max"), 0.15)
    text_thr = _normalize_text_match_thresholds(
        pipe_settings.get("text_match_thresholds")
    )
    if not ocr_text_present:
        # Нет текста этикетки — не гоняем text-match (exclusive / XGB / CrEnc / fin*).
        use_xgb = False
        use_crenc = False
        use_openai_txt = False
        use_fin1 = False
        use_fin2 = False
        _plog("text match skipped: empty OCR")

    exclusive_ids: list[int] = []
    exclusive_seen: set[int] = set()
    for src in (
        status.get("candidates_siglip2_ids") or [],
        status.get("candidates_dinov3_ids") or [],
    ):
        for wid in src:
            i = int(wid)
            if i not in exclusive_seen:
                exclusive_seen.add(i)
                exclusive_ids.append(i)
    exclusive_wines: list[dict[str, Any]] = []
    if exclusive_ids:
        rows_ex = list(
            db.scalars(select(Wine).where(Wine.id.in_(exclusive_ids))).all()
        )
        by_ex = {int(w.id): w for w in rows_ex}
        for wid in exclusive_ids:
            w = by_ex.get(wid)
            if w is None:
                continue
            exclusive_wines.append(
                enrich_wine_dict_for_xgb(
                    {
                        "id": int(w.id),
                        "name": w.name,
                        "label": w.label,
                        "category": w.category,
                        "color": w.color,
                        "grape_variety": w.grape_variety,
                        "region": w.region,
                        "winery": w.winery,
                        "cupage": _cupage_from_label_ocr(w.label_ocr),
                        "label_ocr": w.label_ocr
                        if isinstance(w.label_ocr, dict)
                        else None,
                    }
                )
            )
    cosine_by_id_xgb: dict[int, float] = {}
    for wid_s, v in (status.get("candidates_cosine") or {}).items():
        try:
            cosine_by_id_xgb[int(wid_s)] = float(v)
        except (TypeError, ValueError):
            continue

    # Exclusive lexicon — после XGB; пока только HSV + label-text.
    exclusive_lexicon_reject_ids: set[int] = set()
    exclusive_reject_ids: set[int] = set()
    hsv_reject_ids = {
        int(x) for x in (status.get("hsv_rejected_ids") or [])
    }
    if hsv_reject_ids:
        exclusive_reject_ids |= hsv_reject_ids
    hsv_hard_ids = {
        int(x)
        for x in (
            ((status.get("steps") or {}).get("hsv") or {}).get("hard_rejected_ids")
            or []
        )
    }
    if hsv_hard_ids:
        exclusive_reject_ids |= hsv_hard_ids

    label_text_hr = evaluate_label_text_hard_reject(
        xgb_ocr_text if ocr_text_present else "",
        exclusive_wines,
    )
    status["steps"]["label_text_hard_reject"] = label_text_hr
    label_text_reject_reasons = {
        int(k): str(v)
        for k, v in (label_text_hr.get("reasons") or {}).items()
    }
    if label_text_reject_reasons:
        exclusive_reject_ids |= set(label_text_reject_reasons.keys())
    _plog(
        "label_text_hard_reject",
        rejected=label_text_hr.get("reject_count"),
        ms=label_text_hr.get("ms"),
        xgb_hard_reject=len(exclusive_reject_ids),
    )

    # 2) XGB / CrEnc — параллельно (exclusive lexicon ещё не считали)
    def _run_xgb_scores() -> tuple[list[dict[str, Any]], float, str | None]:
        t0 = time.perf_counter()
        if not use_xgb:
            return [], 0.0, None
        try:
            scored = score_candidates_skipping_hard_reject(
                xgb_ocr_text,
                exclusive_wines,
                cosine_by_id_xgb,
                exclusive_reject_ids,
            )
            return scored, round((time.perf_counter() - t0) * 1000, 1), None
        except Exception as exc:  # noqa: BLE001
            return [], round((time.perf_counter() - t0) * 1000, 1), str(exc)

    def _run_crenc_scores() -> tuple[
        list[dict[str, Any]], float, str | None, float | None
    ]:
        t0 = time.perf_counter()
        if not use_crenc:
            return [], 0.0, None, None
        try:
            if use_crenc_srv:
                scored, thr = score_crenc_candidates_server(
                    xgb_ocr_text, exclusive_wines
                )
                return (
                    scored,
                    round((time.perf_counter() - t0) * 1000, 1),
                    None,
                    float(thr),
                )
            scored = score_crenc_candidates(
                xgb_ocr_text, exclusive_wines
            )
            return scored, round((time.perf_counter() - t0) * 1000, 1), None, None
        except Exception as exc:  # noqa: BLE001
            return [], round((time.perf_counter() - t0) * 1000, 1), str(exc), None

    xgb_scored: list[dict[str, Any]]
    crenc_scored: list[dict[str, Any]]
    xgb_compute_ms = 0.0
    crenc_compute_ms = 0.0
    xgb_score_error: str | None = None
    crenc_score_error: str | None = None
    crenc_server_threshold: float | None = None
    t_xgb = time.perf_counter()
    t_crenc = t_xgb
    n_workers = sum(1 for flag in (use_xgb, use_crenc) if flag) or 1
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=n_workers, thread_name_prefix="xgb_crenc"
    ) as ex_pool:
        fut_xgb = ex_pool.submit(_run_xgb_scores) if use_xgb else None
        fut_crenc = ex_pool.submit(_run_crenc_scores) if use_crenc else None
        if fut_xgb is None:
            xgb_scored = []
        else:
            xgb_scored, xgb_compute_ms, xgb_score_error = fut_xgb.result()
        if fut_crenc is None:
            crenc_scored = []
        else:
            (
                crenc_scored,
                crenc_compute_ms,
                crenc_score_error,
                crenc_server_threshold,
            ) = fut_crenc.result()

    def _make_xgb_step(
        scored: list[dict[str, Any]],
        reject_ids: set[int],
        *,
        compute_ms: float,
        error: str | None,
    ) -> dict[str, Any]:
        xgb_thr = text_thr.get("xgb") or {"match": 0.6, "similar": 0.2}
        xgb_match_t = float(xgb_thr["match"])
        xgb_similar_t = float(xgb_thr["similar"])
        xgb_fw = dict(pipe_settings.get("final_weights") or {}) or None
        if error:
            return {
                "ok": False,
                "error": error,
                "ms": compute_ms,
                "scores": [],
                "by_id": {},
                "threshold": xgb_match_t,
                "thresholds": {"match": xgb_match_t, "similar": xgb_similar_t},
            }
        ranked = apply_xgb_fin(
            scored,
            match_threshold=xgb_match_t,
            similar_threshold=xgb_similar_t,
            exclusive_reject_ids=reject_ids,
            label_text_reject_reasons=label_text_reject_reasons,
            final_weights=xgb_fw,
            cosine_by_id=cosine_by_id_xgb,
        )
        ranked_sorted = sorted(
            ranked,
            key=lambda r: (
                -float(r.get("xgb_fin") or 0),
                -float(r.get("xgb_score") or 0),
            ),
        )
        best_fin = next(
            (
                r
                for r in ranked_sorted
                if r.get("suitable") and float(r.get("xgb_fin") or 0) > 0
            ),
            None,
        )
        n_rejected = sum(1 for r in ranked if r.get("exclusive_rejected"))
        return {
            "ok": True,
            "ms": compute_ms,
            "threshold": xgb_match_t,
            "thresholds": {"match": xgb_match_t, "similar": xgb_similar_t},
            "n_candidates": len(ranked),
            "n_scored": len(ranked),
            "n_hard_rejected": n_rejected,
            "n_suitable": sum(1 for r in ranked if r.get("suitable")),
            "ocr_chars": len(xgb_ocr_text or ""),
            "scores": ranked_sorted,
            "by_id": {str(int(r["id"])): r for r in ranked},
            "best_xgb_fin": best_fin,
            "explain": {
                "text_score": (
                    "xgb_score = P(match) XGBoost (TextScore v1.4 + siglip_cosine); "
                    "считается всегда, в т.ч. при hard reject"
                ),
                "features": (
                    "OCR↔wines.label only (+ label_ocr color/type/vintage); "
                    "no CMS name/winery/grape/region; digit+homoglyphs; "
                    "no cosine/rel/hard_reject in model features"
                ),
                "order": (
                    "HSV → label_text hard_reject → XGB → exclusive lexicon "
                    "on all top-N (no XGB>0.1 gate); "
                    "if max xgb_score < xgb_dead_max → Soft TF-IDF "
                    "(всегда, даже если fin2 выкл) / cosine fallback "
                    "(fin=0 if reject; score always real)"
                ),
                "xgb_fin": (
                    "match|similar → w_ocr×XGB + w_emb×Cosine; "
                    "below similar / hard reject → xgb_fin=0 "
                    "(xgb_fin_pre_reject = без reject); "
                    "winner only from match band; "
                    "dead-XGB → Soft TF-IDF (forced) or cosine"
                ),
                "label_text_hard_reject": (
                    "no shared tokens; OCR≠empty & catalog empty; "
                    "OCR empty & catalog has text → hard_reject "
                    "(XGB score всё равно считается)"
                ),
                "thresholds": {
                    "match": xgb_match_t,
                    "similar": xgb_similar_t,
                },
            },
        }

    def _attach_reuse_xgb(step: dict[str, Any]) -> dict[str, Any]:
        """Досчитать XGB для reuse-hits (OCR↔previous_search_ocr)."""
        if not step.get("ok"):
            return step
        cand = status.get("steps", {}).get("candidates")
        if not isinstance(cand, dict):
            return step
        xgb_thr_r = text_thr.get("xgb") or {"match": 0.6, "similar": 0.2}
        wines_by_id = {
            int(w["id"]): w for w in exclusive_wines if w.get("id") is not None
        }
        by_reuse = score_reuse_previous_search_xgb(
            ocr_text=xgb_ocr_text,
            candidates=cand,
            wines_by_id=wines_by_id,
            cosine_by_id=cosine_by_id_xgb,
            match_threshold=float(xgb_thr_r["match"]),
            similar_threshold=float(xgb_thr_r["similar"]),
            final_weights=dict(pipe_settings.get("final_weights") or {}) or None,
        )
        if not by_reuse:
            return step
        step = dict(step)
        step["by_reuse_sid"] = by_reuse
        scores = list(step.get("scores") or [])
        scores.extend(by_reuse.values())
        scores.sort(
            key=lambda r: (
                -float(r.get("xgb_fin") or 0),
                -float(r.get("xgb_score") or 0),
            )
        )
        step["scores"] = scores
        best_fin = next(
            (
                r
                for r in scores
                if r.get("suitable") and float(r.get("xgb_fin") or 0) > 0
            ),
            None,
        )
        if best_fin is not None:
            step["best_xgb_fin"] = best_fin
        step["n_reuse_scored"] = len(by_reuse)
        return step

    if not use_xgb:
        xgb_step = {
            "ok": False,
            "skipped": True,
            "reason": "text_match_methods",
            "ms": 0,
            "scores": [],
            "by_id": {},
        }
    else:
        xgb_step = _attach_reuse_xgb(
            _make_xgb_step(
                xgb_scored,
                exclusive_reject_ids,
                compute_ms=xgb_compute_ms,
                error=xgb_score_error,
            )
        )
    status["steps"]["xgb_match"] = xgb_step
    cand_step = status.get("steps", {}).get("candidates")
    if isinstance(cand_step, dict):
        attach_xgb_to_candidate_items(cand_step, xgb_step)
        status["steps"]["candidates"] = cand_step
        candidates = cand_step
    if isinstance(xgb_step.get("best_xgb_fin"), dict):
        status["xgb_best_wine_id"] = xgb_step["best_xgb_fin"].get("id")
        status["xgb_best_fin"] = xgb_step["best_xgb_fin"].get("xgb_fin")
    _time_set("xgb_match", t_xgb, duration_ms=xgb_step.get("ms"))
    _plog(
        "xgb_match done",
        ok=xgb_step.get("ok"),
        n=xgb_step.get("n_candidates"),
        scored=xgb_step.get("n_scored"),
        n_hard_rejected=xgb_step.get("n_hard_rejected"),
        suitable=xgb_step.get("n_suitable"),
        best=(xgb_step.get("best_xgb_fin") or {}).get("id")
        if isinstance(xgb_step.get("best_xgb_fin"), dict)
        else None,
        ms=xgb_step.get("ms"),
        error=xgb_step.get("error"),
    )

    # 3) exclusive lexicon — все visual-кандидаты (top-N), без XGB-гейта.
    # Гейт XGB>0.1 обнулял checked при слабом XGB (напр. max≈0.08) и
    # пропускал заведомо чужие этикетки (rose vs «рубин»).
    def _run_exclusive(cands: list[dict[str, Any]]) -> dict[str, Any]:
        if not ocr_text_present:
            return {
                "ok": True,
                "skipped": True,
                "reason": "empty_ocr",
                "rejected_ids": [],
                "reject_count": 0,
                "ms": 0,
            }
        q = str(xgb_ocr_text or exclusive_query).strip()
        return evaluate_exclusive_lexicon(
            q,
            cands,
            use_translit=bool(pipe_settings.get("exclusive_use_translit", True)),
            match_spaced=bool(pipe_settings.get("exclusive_match_spaced", True)),
        )

    gated_wines = list(exclusive_wines)
    gated_ids = [int(w["id"]) for w in gated_wines]

    t_exclusive = time.perf_counter()
    try:
        exclusive_step = _run_exclusive(gated_wines)
    except Exception as exc:  # noqa: BLE001
        exclusive_step = {"ok": False, "error": str(exc), "rejected_ids": []}
    if not isinstance(exclusive_step, dict):
        exclusive_step = {"ok": False, "error": "bad exclusive result"}
    exclusive_step["gate"] = {
        "mode": "all_candidates",
        "n_candidates": len(exclusive_wines),
        "n_gated": len(gated_wines),
        "gated_ids": gated_ids,
    }
    status["steps"]["exclusive_lexicon"] = exclusive_step
    _time_set(
        "exclusive_lexicon",
        t_exclusive,
        duration_ms=exclusive_step.get("ms"),
    )
    _plog(
        "exclusive_lexicon done",
        rejected=exclusive_step.get("reject_count"),
        gated=len(gated_wines),
        of=len(exclusive_wines),
        q_cat=list((exclusive_step.get("query_hits") or {}).get("category") or {}),
        q_type=list((exclusive_step.get("query_hits") or {}).get("type") or {}),
        q_grape=list((exclusive_step.get("query_hits") or {}).get("grape") or {}),
        q_winery=list((exclusive_step.get("query_hits") or {}).get("winery") or {}),
    )
    exclusive_lexicon_reject_ids = {
        int(x) for x in (exclusive_step.get("rejected_ids") or [])
    }
    exclusive_reject_ids |= exclusive_lexicon_reject_ids

    # Пересчитать XGB_fin с exclusive lexicon reject
    if use_xgb and xgb_step.get("ok") and not xgb_score_error:
        xgb_step = _attach_reuse_xgb(
            _make_xgb_step(
                xgb_scored,
                exclusive_reject_ids,
                compute_ms=xgb_compute_ms,
                error=None,
            )
        )
        status["steps"]["xgb_match"] = xgb_step
        cand_step = status.get("steps", {}).get("candidates")
        if isinstance(cand_step, dict):
            attach_xgb_to_candidate_items(cand_step, xgb_step)
            status["steps"]["candidates"] = cand_step
            candidates = cand_step
        if isinstance(xgb_step.get("best_xgb_fin"), dict):
            status["xgb_best_wine_id"] = xgb_step["best_xgb_fin"].get("id")
            status["xgb_best_fin"] = xgb_step["best_xgb_fin"].get("xgb_fin")
        else:
            status["xgb_best_wine_id"] = None
            status["xgb_best_fin"] = None

    # Bypass hard reject when cos ≥ thr AND xgb_score ≥ thr.
    hard_reject_bypassed_ids: list[int] = []
    if (
        hard_reject_ignore_high_scores
        and exclusive_reject_ids
        and use_xgb
        and isinstance(xgb_step, dict)
        and xgb_step.get("ok")
        and not xgb_score_error
    ):
        cos_thr = float(hard_reject_ignore_cosine_min)
        xgb_thr = float(hard_reject_ignore_xgb_min)
        by_id_hr = xgb_step.get("by_id") or {}
        by_reuse_hr = xgb_step.get("by_reuse_sid") or {}
        xgb_score_map: dict[int, float] = {}
        if isinstance(by_id_hr, dict):
            for k, v in by_id_hr.items():
                if not isinstance(v, dict) or v.get("xgb_score") is None:
                    continue
                try:
                    xgb_score_map[int(k)] = float(v.get("xgb_score"))
                except (TypeError, ValueError):
                    continue
        if isinstance(by_reuse_hr, dict):
            for rv in by_reuse_hr.values():
                if not isinstance(rv, dict) or rv.get("id") is None:
                    continue
                if rv.get("xgb_score") is None:
                    continue
                try:
                    wid_r = int(rv["id"])
                    sc_r = float(rv.get("xgb_score"))
                except (TypeError, ValueError, KeyError):
                    continue
                prev_sc = xgb_score_map.get(wid_r)
                if prev_sc is None or sc_r > prev_sc:
                    xgb_score_map[wid_r] = sc_r
        for wid_hr in list(exclusive_reject_ids):
            try:
                raw_cos = cosine_by_id_xgb.get(int(wid_hr))
                cos_v = float(raw_cos) if raw_cos is not None else None
            except (TypeError, ValueError):
                cos_v = None
            xgb_v = xgb_score_map.get(int(wid_hr))
            if (
                cos_v is not None
                and xgb_v is not None
                and cos_v >= cos_thr
                and xgb_v >= xgb_thr
            ):
                exclusive_reject_ids.discard(int(wid_hr))
                exclusive_lexicon_reject_ids.discard(int(wid_hr))
                hard_reject_bypassed_ids.append(int(wid_hr))
        hard_reject_bypassed_ids = sorted(set(hard_reject_bypassed_ids))
        if hard_reject_bypassed_ids:
            # Mark candidates for UI / debug
            for branch in ("siglip2", "dinov3"):
                items = (candidates.get(branch) or {}).get("items") or []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    try:
                        if int(it.get("id") or 0) in set(hard_reject_bypassed_ids):
                            it["hard_reject_bypassed"] = True
                            it["hsv_rejected"] = False
                            it["hsv_suitable"] = True
                            it["hsv_hard_rejected"] = False
                    except (TypeError, ValueError):
                        continue
            # HSV status lists — drop bypassed from soft/hard reject sets
            hsv_step_now = (status.get("steps") or {}).get("hsv") or {}
            if isinstance(hsv_step_now, dict):
                bypass_set = set(hard_reject_bypassed_ids)
                for key in ("rejected_ids", "hard_rejected_ids"):
                    raw_ids = hsv_step_now.get(key) or []
                    if isinstance(raw_ids, list):
                        hsv_step_now[key] = [
                            int(x) for x in raw_ids if int(x) not in bypass_set
                        ]
                kept = list(hsv_step_now.get("kept_by_cosine_ids") or [])
                for wid_b in hard_reject_bypassed_ids:
                    if wid_b not in kept:
                        kept.append(wid_b)
                hsv_step_now["kept_by_cosine_ids"] = sorted(set(int(x) for x in kept))
                hsv_step_now["hard_reject_bypassed_ids"] = hard_reject_bypassed_ids
                status["steps"]["hsv"] = hsv_step_now
                status["hsv_rejected_ids"] = list(hsv_step_now.get("rejected_ids") or [])
            # Re-score XGB_fin without bypassed rejects
            xgb_step = _attach_reuse_xgb(
                _make_xgb_step(
                    xgb_scored,
                    exclusive_reject_ids,
                    compute_ms=xgb_compute_ms,
                    error=None,
                )
            )
            status["steps"]["xgb_match"] = xgb_step
            cand_step = status.get("steps", {}).get("candidates")
            if isinstance(cand_step, dict):
                attach_xgb_to_candidate_items(cand_step, xgb_step)
                status["steps"]["candidates"] = cand_step
                candidates = cand_step
            if isinstance(xgb_step.get("best_xgb_fin"), dict):
                status["xgb_best_wine_id"] = xgb_step["best_xgb_fin"].get("id")
                status["xgb_best_fin"] = xgb_step["best_xgb_fin"].get("xgb_fin")
            else:
                status["xgb_best_wine_id"] = None
                status["xgb_best_fin"] = None
    status["steps"]["hard_reject_bypass"] = {
        "enabled": bool(hard_reject_ignore_high_scores),
        "cosine_min": float(hard_reject_ignore_cosine_min),
        "xgb_min": float(hard_reject_ignore_xgb_min),
        "bypassed_ids": hard_reject_bypassed_ids,
        "n": len(hard_reject_bypassed_ids),
        "rule": (
            f"cos ≥ {float(hard_reject_ignore_cosine_min):.2f} and "
            f"xgb_score ≥ {float(hard_reject_ignore_xgb_min):.2f} "
            "→ ignore hard reject (HSV/exclusive/label-text)"
        ),
    }
    if hard_reject_bypassed_ids:
        _plog(
            "hard_reject_bypass",
            n=len(hard_reject_bypassed_ids),
            ids=hard_reject_bypassed_ids,
            cos_min=hard_reject_ignore_cosine_min,
            xgb_min=hard_reject_ignore_xgb_min,
        )

    if not use_crenc:
        crenc_step: dict[str, Any] = {
            "ok": False,
            "skipped": True,
            "reason": "text_match_methods",
            "ms": 0,
            "scores": [],
            "by_id": {},
            "backend": crenc_backend,
        }
    else:
        crenc_thr_key = "crenc_srv" if use_crenc_srv else "crenc"
        crenc_thr = text_thr.get(crenc_thr_key) or text_thr.get("crenc") or {
            "match": 0.55,
            "similar": 0.25,
        }
        crenc_match_t = float(crenc_thr["match"])
        crenc_similar_t = float(crenc_thr["similar"])
        crenc_fw = dict(pipe_settings.get("final_weights") or {}) or None
        if crenc_score_error:
            crenc_step = {
                "ok": False,
                "error": crenc_score_error,
                "ms": crenc_compute_ms,
                "scores": [],
                "by_id": {},
                "threshold": crenc_match_t,
                "thresholds": {
                    "match": crenc_match_t,
                    "similar": crenc_similar_t,
                },
                "backend": crenc_backend,
            }
        else:
            ranked_c = apply_crenc_fin(
                crenc_scored,
                match_threshold=crenc_match_t,
                similar_threshold=crenc_similar_t,
                exclusive_reject_ids=exclusive_reject_ids,
                final_weights=crenc_fw,
                cosine_by_id=cosine_by_id_xgb,
            )
            ranked_c_sorted = sorted(
                ranked_c,
                key=lambda r: (
                    -float(r.get("crenc_fin") or 0),
                    -float(r.get("crenc_score") or 0),
                ),
            )
            best_c = next(
                (
                    r
                    for r in ranked_c_sorted
                    if r.get("suitable") and float(r.get("crenc_fin") or 0) > 0
                ),
                None,
            )
            crenc_step = {
                "ok": True,
                "ms": crenc_compute_ms,
                "threshold": crenc_match_t,
                "thresholds": {
                    "match": crenc_match_t,
                    "similar": crenc_similar_t,
                },
                "backend": crenc_backend,
                "n_candidates": len(ranked_c),
                "n_suitable": sum(1 for r in ranked_c if r.get("suitable")),
                "ocr_chars": len(xgb_ocr_text or ""),
                "scores": ranked_c_sorted,
                "by_id": {str(int(r["id"])): r for r in ranked_c},
                "best_crenc_fin": best_c,
                "explain": {
                    "text_score": "crenc_score = P(match) CrossEncoder (TextScore)",
                    "crenc_fin": (
                        "match|similar → w_ocr×CrEnc + w_emb×Cosine; "
                        "below similar or exclusive → 0; "
                        "winner only from match band"
                    ),
                    "thresholds": {
                        "match": crenc_match_t,
                        "similar": crenc_similar_t,
                    },
                },
            }
    status["steps"]["crenc_match"] = crenc_step
    cand_step = status.get("steps", {}).get("candidates")
    if isinstance(cand_step, dict):
        attach_crenc_to_candidate_items(cand_step, crenc_step)
        status["steps"]["candidates"] = cand_step
        candidates = cand_step
    if isinstance(crenc_step.get("best_crenc_fin"), dict):
        status["crenc_best_wine_id"] = crenc_step["best_crenc_fin"].get("id")
        status["crenc_best_fin"] = crenc_step["best_crenc_fin"].get("crenc_fin")
    crenc_timing_key = (
        "crenc_match_srv" if use_crenc_srv else "crenc_match_loc"
    )
    if use_crenc:
        _time_set(crenc_timing_key, t_crenc, duration_ms=crenc_step.get("ms"))
    _plog(
        "crenc_match done",
        ok=crenc_step.get("ok"),
        skipped=crenc_step.get("skipped"),
        backend=crenc_step.get("backend"),
        n=crenc_step.get("n_candidates"),
        best=(crenc_step.get("best_crenc_fin") or {}).get("id")
        if isinstance(crenc_step.get("best_crenc_fin"), dict)
        else None,
        ms=crenc_step.get("ms"),
        error=crenc_step.get("error"),
    )

    # OpenAI сравнение текста (survivors exclusive lexicon) — после exclusive
    t_openai_txt = time.perf_counter()
    if not use_openai_txt:
        openai_txt_step: dict[str, Any] = {
            "ok": False,
            "skipped": True,
            "reason": "text_match_methods",
            "ms": 0,
            "scores": [],
            "by_id": {},
        }
    else:
        survivors = [
            w
            for w in exclusive_wines
            if int(w["id"]) not in exclusive_lexicon_reject_ids
        ]
        try:
            openai_txt_step = evaluate_openai_txt_match(
                str(xgb_ocr_text or ""),
                survivors,
            )
        except Exception as exc:  # noqa: BLE001
            openai_txt_step = {
                "ok": False,
                "error": str(exc),
                "ms": round((time.perf_counter() - t_openai_txt) * 1000, 1),
                "scores": [],
                "by_id": {},
            }
        if not isinstance(openai_txt_step, dict):
            openai_txt_step = {
                "ok": False,
                "error": "bad openai_txt result",
                "ms": 0,
                "scores": [],
                "by_id": {},
            }
    openai_txt_step["exclusive_lexicon_rejected"] = len(
        exclusive_lexicon_reject_ids
    )
    openai_txt_step["n_survivors_sent"] = openai_txt_step.get("n_input") or openai_txt_step.get(
        "candidates_sent"
    )
    status["steps"]["openai_txt_match"] = openai_txt_step
    cand_step = status.get("steps", {}).get("candidates")
    if isinstance(cand_step, dict):
        attach_openai_txt_to_candidate_items(cand_step, openai_txt_step)
        status["steps"]["candidates"] = cand_step
        candidates = cand_step
    if use_openai_txt:
        _time_set(
            "openai_txt_match",
            t_openai_txt,
            duration_ms=openai_txt_step.get("ms"),
        )
    _plog(
        "openai_txt_match done",
        ok=openai_txt_step.get("ok"),
        skipped=openai_txt_step.get("skipped"),
        n_input=openai_txt_step.get("n_input"),
        n_scored=openai_txt_step.get("n_scored"),
        best=(openai_txt_step.get("best") or {}).get("id")
        if isinstance(openai_txt_step.get("best"), dict)
        else None,
        ms=openai_txt_step.get("ms"),
        error=openai_txt_step.get("error"),
    )
    _save_status(db, row, status)

    # 4b) OCR TextScore vs unique embedding candidates (SigLIP2 ∪ DINOv3)
    t0 = time.perf_counter()
    visual_ids: list[int] = []
    seen_ids: set[int] = set()
    for src in (
        status.get("candidates_siglip2_ids") or [],
        status.get("candidates_dinov3_ids") or [],
    ):
        for wid in src:
            i = int(wid)
            if i not in seen_ids:
                seen_ids.add(i)
                visual_ids.append(i)
    _plog(">> ocr_wine_id", visual_n=len(visual_ids), ocr_text=ocr_text_present)
    cosine_by_id: dict[int, float] = {}
    for wid_s, v in (status.get("candidates_cosine") or {}).items():
        try:
            cosine_by_id[int(wid_s)] = float(v)
        except (TypeError, ValueError):
            continue
    for src_items in (
        (candidates.get("siglip2") or {}).get("items") or [],
        (candidates.get("dinov3") or {}).get("items") or [],
    ):
        for it in src_items:
            if not isinstance(it, dict) or it.get("id") is None:
                continue
            wid = int(it["id"])
            cos = it.get("cosine_similarity")
            if cos is None:
                continue
            try:
                cos_f = float(cos)
            except (TypeError, ValueError):
                continue
            prev = cosine_by_id.get(wid)
            if prev is None or cos_f > prev:
                cosine_by_id[wid] = cos_f
    if not ocr_text_present:
        status["steps"]["ocr_wine_id"] = {
            "ok": False,
            "skipped": True,
            "reason": "empty_ocr",
        }
        _plog("ocr_wine_id skipped: empty OCR")
    else:
        try:
            ocr_help = evaluate_ocr_wine_id_help(
                db,
                ocr_step=status.get("steps", {}).get("ocr") or {},
                visual_candidate_ids=visual_ids,
                cosine_by_id=cosine_by_id,
                top_k=max(20, len(visual_ids)),
                exclusive_reject_ids=exclusive_reject_ids,
                compute_fin1=use_fin1,
                # Soft TF-IDF нужен и для dead-XGB fallback, даже если fin2
                # выключен в text_match_methods.
                compute_fin2=use_fin2 or (final_method == "xgb"),
            )
            status["steps"]["ocr_wine_id"] = ocr_help
            top_id = None
            if isinstance(ocr_help, dict):
                top_id = ocr_help.get("best_wine_id") or ocr_help.get("wine_id")
                ranked = (
                    ocr_help.get("final_ranked")
                    or ocr_help.get("ranked")
                    or ocr_help.get("items")
                    or []
                )
                if top_id is None and ranked and isinstance(ranked[0], dict):
                    top_id = ranked[0].get("wine_id") or ranked[0].get("id")
            _plog(
                "ocr_wine_id",
                ok=ocr_help.get("ok") if isinstance(ocr_help, dict) else None,
                top=top_id,
                final=(
                    (ocr_help.get("best_final") or {}).get("final_score")
                    if isinstance(ocr_help, dict)
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            status["steps"]["ocr_wine_id"] = {"ok": False, "error": str(exc)}
            _plog("FAIL ocr_wine_id failed", error=str(exc))
    _time_set("ocr_wine_id", t0)

    # Per-OCR finals for selected text method (cards / sort / decision primary)
    final_by_ocr_step: dict[str, Any] = {
        "ok": True,
        "method": final_method if final_method in text_methods else None,
        "prefer": final_ocr_pref,
        "primary": final_ocr_engine,
        "engines": sorted(ocr_texts_by_engine.keys()),
        "by_ocr": {},
    }
    method_for_ocr = final_by_ocr_step["method"]
    by_ocr_payload: dict[str, Any] = {}
    ocr_help = status.get("steps", {}).get("ocr_wine_id") or {}

    def _pack_by_id(
        fin_map: dict[str, float],
        text_map: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        best_id = None
        best_sc = -1.0
        for wid_s, sc in fin_map.items():
            try:
                v = float(sc)
            except (TypeError, ValueError):
                continue
            if v > best_sc:
                best_sc = v
                try:
                    best_id = int(wid_s)
                except (TypeError, ValueError):
                    best_id = None
        out: dict[str, Any] = {
            "by_id": {k: round(float(v), 6) for k, v in fin_map.items()},
            "best_id": best_id,
            "best_score": round(best_sc, 6) if best_sc >= 0 else None,
        }
        if text_map is not None:
            out["text_by_id"] = {
                k: round(float(v), 6) for k, v in text_map.items()
            }
        return out

    if method_for_ocr in ("fin1", "fin2") and isinstance(ocr_help, dict):
        score_key = "final_score" if method_for_ocr == "fin1" else "final_score2"
        pv = ocr_help.get("per_variant") or {}
        if not isinstance(pv, dict):
            pv = {}

        def _fin_channel_key(raw_key: str) -> str | None:
            k = str(raw_key or "").strip()
            if not k:
                return None
            if k in ocr_texts_by_engine or k in (
                "google_vision",
                "gemini",
                "openai",
                "yandex",
                "deepseek",
                "qwen",
            ):
                return k
            m = re.match(r"^([A-D])(?:_|$)", k)
            return m.group(1) if m else None

        # All per_variant channels (letters + cloud), not only Final OCR engines
        for raw_key, entry in pv.items():
            ch = _fin_channel_key(str(raw_key))
            if not ch or not isinstance(entry, dict):
                continue
            fin_m: dict[str, float] = {}
            txt_m: dict[str, float] = {}
            for row in entry.get("visual_support") or []:
                if not isinstance(row, dict) or row.get("id") is None:
                    continue
                wid_s = str(int(row["id"]))
                raw_fin = row.get(score_key)
                if raw_fin is None and method_for_ocr == "fin1":
                    # compute from score_01 if final missing
                    try:
                        from app.pipeline.ocr_match import compute_final_score

                        cos = cosine_by_id_xgb.get(int(row["id"]))
                        mixed = compute_final_score(
                            text_score_01=float(row.get("score_01") or 0),
                            cosine=cos,
                        )
                        raw_fin = mixed.get("final_score")
                    except Exception:  # noqa: BLE001
                        raw_fin = row.get("score_01")
                try:
                    fin_v = float(raw_fin)
                except (TypeError, ValueError):
                    continue
                prev = fin_m.get(wid_s)
                if prev is None or fin_v > prev:
                    fin_m[wid_s] = fin_v
                try:
                    txt_v = float(
                        row.get("score_01")
                        if method_for_ocr == "fin1"
                        else row.get("score2_01") or 0
                    )
                    prev_t = txt_m.get(wid_s)
                    if prev_t is None or txt_v > prev_t:
                        txt_m[wid_s] = txt_v
                except (TypeError, ValueError):
                    pass
            if not fin_m:
                continue
            existing = by_ocr_payload.get(ch)
            if isinstance(existing, dict) and isinstance(existing.get("by_id"), dict):
                merged_fin = dict(existing["by_id"])
                for k, v in fin_m.items():
                    if k not in merged_fin or float(v) > float(merged_fin[k]):
                        merged_fin[k] = v
                merged_txt = dict(existing.get("text_by_id") or {})
                for k, v in txt_m.items():
                    if k not in merged_txt or float(v) > float(merged_txt[k]):
                        merged_txt[k] = v
                by_ocr_payload[ch] = _pack_by_id(merged_fin, merged_txt or None)
            else:
                by_ocr_payload[ch] = _pack_by_id(fin_m, txt_m or None)
    elif method_for_ocr == "xgb" and use_xgb and exclusive_wines:
        xgb_thr = text_thr.get("xgb") or {"match": 0.6, "similar": 0.2}
        xgb_match_t = float(xgb_thr["match"])
        xgb_similar_t = float(xgb_thr["similar"])
        xgb_fw = dict(pipe_settings.get("final_weights") or {}) or None
        primary_by = (
            (status.get("steps") or {}).get("xgb_match") or {}
        ).get("by_id") or {}
        for eng, text in ocr_texts_by_engine.items():
            if (
                eng == final_ocr_engine
                and isinstance(primary_by, dict)
                and primary_by
            ):
                fin_m = {}
                txt_m = {}
                for k, v in primary_by.items():
                    if not isinstance(v, dict):
                        continue
                    try:
                        fin_m[str(int(k))] = float(v.get("xgb_fin") or 0)
                        txt_m[str(int(k))] = float(v.get("xgb_score") or 0)
                    except (TypeError, ValueError):
                        continue
                if fin_m:
                    by_ocr_payload[eng] = _pack_by_id(fin_m, txt_m)
                continue
            try:
                scored = score_candidates_skipping_hard_reject(
                    text,
                    exclusive_wines,
                    cosine_by_id_xgb,
                    exclusive_reject_ids,
                )
                ranked = apply_xgb_fin(
                    scored,
                    match_threshold=xgb_match_t,
                    similar_threshold=xgb_similar_t,
                    exclusive_reject_ids=exclusive_reject_ids,
                    final_weights=xgb_fw,
                    cosine_by_id=cosine_by_id_xgb,
                )
                fin_m = {
                    str(int(r["id"])): float(r.get("xgb_fin") or 0) for r in ranked
                }
                txt_m = {
                    str(int(r["id"])): float(r.get("xgb_score") or 0) for r in ranked
                }
                by_ocr_payload[eng] = _pack_by_id(fin_m, txt_m)
            except Exception as exc:  # noqa: BLE001
                by_ocr_payload[eng] = {"error": str(exc), "by_id": {}}
    elif method_for_ocr in ("crenc", "crenc_srv") and use_crenc and exclusive_wines:
        crenc_thr = text_thr.get(method_for_ocr) or text_thr.get("crenc") or {
            "match": 0.6,
            "similar": 0.2,
        }
        crenc_match_t = float(crenc_thr["match"])
        crenc_similar_t = float(crenc_thr["similar"])
        crenc_fw = dict(pipe_settings.get("final_weights") or {}) or None
        primary_by = (
            (status.get("steps") or {}).get("crenc_match") or {}
        ).get("by_id") or {}
        for eng, text in ocr_texts_by_engine.items():
            if (
                eng == final_ocr_engine
                and isinstance(primary_by, dict)
                and primary_by
            ):
                fin_m = {}
                txt_m = {}
                for k, v in primary_by.items():
                    if not isinstance(v, dict):
                        continue
                    try:
                        fin_m[str(int(k))] = float(v.get("crenc_fin") or 0)
                        txt_m[str(int(k))] = float(v.get("crenc_score") or 0)
                    except (TypeError, ValueError):
                        continue
                if fin_m:
                    by_ocr_payload[eng] = _pack_by_id(fin_m, txt_m)
                continue
            try:
                if method_for_ocr == "crenc_srv":
                    scored_c, _thr = score_crenc_candidates_server(
                        text, exclusive_wines
                    )
                else:
                    scored_c = score_crenc_candidates(
                        text, exclusive_wines
                    )
                ranked_c = apply_crenc_fin(
                    scored_c,
                    match_threshold=crenc_match_t,
                    similar_threshold=crenc_similar_t,
                    exclusive_reject_ids=exclusive_reject_ids,
                    final_weights=crenc_fw,
                    cosine_by_id=cosine_by_id_xgb,
                )
                fin_m = {
                    str(int(r["id"])): float(r.get("crenc_fin") or 0)
                    for r in ranked_c
                }
                txt_m = {
                    str(int(r["id"])): float(r.get("crenc_score") or 0)
                    for r in ranked_c
                }
                by_ocr_payload[eng] = _pack_by_id(fin_m, txt_m)
            except Exception as exc:  # noqa: BLE001
                by_ocr_payload[eng] = {"error": str(exc), "by_id": {}}
    elif method_for_ocr == "openai_txt" and use_openai_txt:
        # Один прогон на final OCR; одинаковые scores во все каналы не дублируем —
        # кладём только в primary engine.
        primary_by = (
            (status.get("steps") or {}).get("openai_txt_match") or {}
        ).get("by_id") or {}
        fin_m: dict[str, float] = {}
        txt_m: dict[str, float] = {}
        if isinstance(primary_by, dict):
            for k, v in primary_by.items():
                if not isinstance(v, dict):
                    continue
                try:
                    wid_s = str(int(k))
                    p = float(v.get("llm_txt") or 0)
                except (TypeError, ValueError):
                    continue
                fin_m[wid_s] = p
                txt_m[wid_s] = p
        if fin_m and final_ocr_engine:
            by_ocr_payload[str(final_ocr_engine)] = _pack_by_id(fin_m, txt_m)
        elif fin_m:
            by_ocr_payload["openai_txt"] = _pack_by_id(fin_m, txt_m)

    final_by_ocr_step["by_ocr"] = by_ocr_payload
    status["steps"]["final_by_ocr"] = final_by_ocr_step
    # Attach to candidate cards
    cand_step_f = status.get("steps", {}).get("candidates")
    if isinstance(cand_step_f, dict):
        for branch in ("siglip2", "dinov3"):
            block = cand_step_f.get(branch) or {}
            items = block.get("items") if isinstance(block, dict) else None
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict) or it.get("id") is None:
                    continue
                wid_s = str(int(it["id"]))
                scores: dict[str, float] = {}
                for eng, pack in by_ocr_payload.items():
                    raw = (pack.get("by_id") or {}).get(wid_s)
                    if raw is None:
                        continue
                    try:
                        scores[eng] = float(raw)
                    except (TypeError, ValueError):
                        continue
                it["final_by_ocr"] = scores
                it["final_ocr_primary"] = final_ocr_engine
        status["steps"]["candidates"] = cand_step_f
        candidates = cand_step_f
    _plog(
        "final_by_ocr",
        method=method_for_ocr,
        primary=final_ocr_engine,
        engines=list(by_ocr_payload.keys()),
    )

    emb_ok = bool(emb_step.get("siglip2", {}).get("ok"))
    if use_dinov3:
        emb_ok = emb_ok and bool(emb_step.get("dinov3", {}).get("ok"))
    ocr_ok = bool(status["steps"].get("ocr", {}).get("ok"))
    status["ok"] = emb_ok  # embeddings — обязательный минимум пайплайна
    if not status["ok"]:
        errs = []
        for k, v in emb_step.items():
            if isinstance(v, dict) and not v.get("ok") and not v.get("skipped"):
                errs.append(f"{k}: {v.get('error')}")
        status["error"] = "; ".join(errs) if errs else "ошибка эмбеддингов"
    elif not ocr_ok:
        status["error"] = None  # OCR soft-fail
        status["warnings"] = ["OCR не выполнен"]

    # Decision: final_score_method + пороги match/similar; пустой OCR → cosine.
    status["matched_wine_id"] = None
    status["matched_wine_confidence"] = None
    status["matched_wine"] = None
    hsv_step_live = status["steps"].get("hsv")
    if isinstance(hsv_step_live, dict):
        hsv_step_live["winner"] = None
    status["similar_wine_ids"] = []
    status["final_score_method"] = final_method
    siglip_items = (candidates.get("siglip2") or {}).get("items") or []
    dino_items = (candidates.get("dinov3") or {}).get("items") or []
    ocr_help = status.get("steps", {}).get("ocr_wine_id") or {}
    best_final = (
        ocr_help.get("best_final") if isinstance(ocr_help, dict) else None
    )
    best_final2 = (
        ocr_help.get("best_final2") if isinstance(ocr_help, dict) else None
    )
    if not use_fin1:
        best_final = None
    if not use_fin2:
        best_final2 = None
    xgb_best = (status.get("steps") or {}).get("xgb_match", {}).get("best_xgb_fin")
    crenc_best = (status.get("steps") or {}).get("crenc_match", {}).get(
        "best_crenc_fin"
    )
    by_cos = {
        int(it["id"]): it
        for it in list(siglip_items) + list(dino_items)
        if isinstance(it, dict) and it.get("id") is not None
    }

    def _attach_card(matched: dict, wine_id: int) -> None:
        card = by_cos.get(int(wine_id)) or {}
        for k in ("slug", "photo_url", "label_url", "label", "hsv", "color_delta"):
            if card.get(k) is not None:
                matched[k] = card.get(k)
        if not matched.get("name") and card.get("name"):
            matched["name"] = card.get("name")

    def _set_matched(wid: int, conf_f: float | None, source: str, extra: dict) -> None:
        status["matched_wine_id"] = wid
        status["matched_wine_confidence"] = conf_f
        status["matched_wine"] = {
            "id": wid,
            "confidence": conf_f,
            "source": source,
            **extra,
        }
        _attach_card(status["matched_wine"], wid)
        hsv_step_now = status["steps"].get("hsv")
        if isinstance(hsv_step_now, dict):
            raw = (hsv_step_now.get("by_id") or {}).get(str(int(wid)))
            if raw is None:
                raw = status["matched_wine"].get("hsv")
            try:
                hsv_step_now["winner"] = (
                    round(float(raw), 4) if raw is not None else None
                )
            except (TypeError, ValueError):
                hsv_step_now["winner"] = None
            if hsv_step_now.get("winner") is not None:
                status["matched_wine"]["hsv"] = hsv_step_now["winner"]
        cd_step_now = status["steps"].get("color_delta")
        if isinstance(cd_step_now, dict):
            raw_cd = (cd_step_now.get("by_id") or {}).get(str(int(wid)))
            if raw_cd is None:
                raw_cd = status["matched_wine"].get("color_delta")
            try:
                cd_step_now["winner"] = (
                    round(float(raw_cd), 2) if raw_cd is not None else None
                )
            except (TypeError, ValueError):
                cd_step_now["winner"] = None
            if cd_step_now.get("winner") is not None:
                status["matched_wine"]["color_delta"] = cd_step_now["winner"]

    def _scores_for_method(method_id: str) -> list[tuple[int, float]]:
        """[(wine_id, score)] for band classification; exclusive → drop.

        Prefer per-OCR scores for resolved final_ocr (primary channel).
        """
        rows: list[tuple[int, float]] = []
        excl = exclusive_reject_ids
        fbo = (status.get("steps") or {}).get("final_by_ocr") or {}
        primary = fbo.get("primary") if isinstance(fbo, dict) else None
        by_ocr = (fbo.get("by_ocr") or {}) if isinstance(fbo, dict) else {}
        primary_pack = (
            by_ocr.get(str(primary)) if primary and isinstance(by_ocr, dict) else None
        )

        def _from_pack(use_text: bool) -> list[tuple[int, float]] | None:
            if not isinstance(primary_pack, dict):
                return None
            src = (
                primary_pack.get("text_by_id")
                if use_text
                else primary_pack.get("by_id")
            ) or {}
            if not isinstance(src, dict) or not src:
                # fin scores live in by_id; for band of fin1 use by_id as score
                src = primary_pack.get("by_id") or {}
            out: list[tuple[int, float]] = []
            for k, v in src.items():
                try:
                    wid = int(k)
                    sc = float(v)
                except (TypeError, ValueError):
                    continue
                if wid in excl:
                    continue
                out.append((wid, sc))
            return out if out else None

        fbo_method = fbo.get("method") if isinstance(fbo, dict) else None

        if method_id == "fin1":
            # Pack только если final_by_ocr именно fin1 (иначе by_id = другой метод).
            if fbo_method == "fin1":
                packed = _from_pack(use_text=False)
                if packed is not None:
                    return packed
            ranked = (
                (ocr_help.get("final_ranked") if isinstance(ocr_help, dict) else None)
                or []
            )
            for r in ranked:
                if not isinstance(r, dict) or r.get("id") is None:
                    continue
                wid = int(r["id"])
                if wid in excl:
                    continue
                try:
                    sc = float(r.get("final_score"))
                except (TypeError, ValueError):
                    continue
                rows.append((wid, sc))
        elif method_id == "fin2":
            if fbo_method == "fin2":
                packed = _from_pack(use_text=False)
                if packed is not None:
                    return packed
            ranked = (
                (ocr_help.get("final_ranked") if isinstance(ocr_help, dict) else None)
                or []
            )
            for r in ranked:
                if not isinstance(r, dict) or r.get("id") is None:
                    continue
                wid = int(r["id"])
                if wid in excl:
                    continue
                raw = r.get("final_score2")
                if raw is None:
                    continue
                try:
                    sc = float(raw)
                except (TypeError, ValueError):
                    continue
                rows.append((wid, sc))
        elif method_id == "xgb":
            # Band by TextScore (xgb_score); prefer primary OCR pack
            packed = _from_pack(use_text=True)
            if packed is not None:
                rows.extend(packed)
            else:
                by_id = ((status.get("steps") or {}).get("xgb_match") or {}).get(
                    "by_id"
                ) or {}
                for k, v in by_id.items():
                    if not isinstance(v, dict):
                        continue
                    try:
                        wid = int(k)
                    except (TypeError, ValueError):
                        continue
                    if wid in excl or v.get("exclusive_rejected"):
                        continue
                    try:
                        sc = float(v.get("xgb_score"))
                    except (TypeError, ValueError):
                        continue
                    rows.append((wid, sc))
            # Reuse-hits: query OCR ↔ previous_search_ocr (не режем exclusive)
            by_reuse = (
                ((status.get("steps") or {}).get("xgb_match") or {}).get(
                    "by_reuse_sid"
                )
                or {}
            )
            if isinstance(by_reuse, dict):
                for v in by_reuse.values():
                    if not isinstance(v, dict) or v.get("id") is None:
                        continue
                    try:
                        wid = int(v["id"])
                        sc = float(v.get("xgb_score"))
                    except (TypeError, ValueError, KeyError):
                        continue
                    rows.append((wid, sc))
        elif method_id in ("crenc", "crenc_srv"):
            packed = _from_pack(use_text=True)
            if packed is not None:
                return packed
            by_id = ((status.get("steps") or {}).get("crenc_match") or {}).get("by_id") or {}
            for k, v in by_id.items():
                if not isinstance(v, dict):
                    continue
                try:
                    wid = int(k)
                except (TypeError, ValueError):
                    continue
                if wid in excl or v.get("exclusive_rejected"):
                    continue
                try:
                    sc = float(v.get("crenc_score"))
                except (TypeError, ValueError):
                    continue
                rows.append((wid, sc))
        elif method_id == "openai_txt":
            packed = _from_pack(use_text=True)
            if packed is not None:
                return packed
            by_id = (
                ((status.get("steps") or {}).get("openai_txt_match") or {}).get(
                    "by_id"
                )
                or {}
            )
            for k, v in by_id.items():
                if not isinstance(v, dict):
                    continue
                try:
                    wid = int(k)
                except (TypeError, ValueError):
                    continue
                if wid in excl:
                    continue
                try:
                    sc = float(v.get("llm_txt"))
                except (TypeError, ValueError):
                    continue
                rows.append((wid, sc))
        return rows

    method = final_method if final_method in text_methods else None
    decision: dict[str, Any] = {
        "ocr_text_present": ocr_text_present,
        "final_score_method": method,
        "empty_ocr_cosine_threshold": empty_ocr_cos_thr,
        "xgb_dead_max": xgb_dead_max,
    }

    if not ocr_text_present:
        # Exact match by max cosine among visual candidates
        # (skip exclusive / HSV / label-text hard-reject).
        best_id = None
        best_cos = -1.0
        for wid, cos in cosine_by_id.items():
            if int(wid) in exclusive_reject_ids:
                continue
            if cos > best_cos:
                best_cos = float(cos)
                best_id = int(wid)
        decision["cosine_best"] = (
            {"id": best_id, "cosine": best_cos} if best_id is not None else None
        )
        if best_id is not None and best_cos >= float(empty_ocr_cos_thr):
            _set_matched(
                best_id,
                best_cos,
                "empty_ocr_cosine",
                {
                    "cosine_similarity": best_cos,
                    "name": by_cos.get(best_id, {}).get("name"),
                    "winery": by_cos.get(best_id, {}).get("winery"),
                },
            )
            decision["band"] = "match"
        else:
            decision["band"] = "none"
            decision["note"] = (
                "empty OCR; max cosine below empty_ocr_cosine_threshold"
                if best_id is not None
                else "empty OCR; no eligible candidates"
            )
    elif method:
        pair = text_thr.get(method) or {"match": 0.55, "similar": 0.25}
        match_t = float(pair["match"])
        similar_t = float(pair["similar"])
        decision["thresholds"] = {"match": match_t, "similar": similar_t}
        scored_rows = _scores_for_method(method)

        # Dead-XGB: all survivors have tiny TextScore → not by XGB_fin.
        xgb_dead = False
        if method == "xgb":
            max_xgb = max((sc for _, sc in scored_rows), default=-1.0)
            decision["max_xgb_score"] = (
                round(float(max_xgb), 6) if max_xgb >= 0 else None
            )
            if scored_rows and float(max_xgb) < float(xgb_dead_max):
                xgb_dead = True
                decision["xgb_dead"] = True

        if xgb_dead:
            status["similar_wine_ids"] = []
            decision["match_ids"] = []
            decision["similar_ids"] = []
            picked = False
            # Soft TF-IDF всегда: даже если fin2 выключен в text_match_methods.
            fin2_pair = text_thr.get("fin2") or {
                "match": 0.55,
                "similar": 0.25,
            }
            fin2_match_t = float(fin2_pair["match"])
            fin2_similar_t = float(fin2_pair["similar"])
            fin2_rows = _scores_for_method("fin2")
            fin2_similar: list[int] = []
            best_fin2: tuple[int, float] | None = None
            for wid, sc in fin2_rows:
                band = score_band(
                    sc, match=fin2_match_t, similar=fin2_similar_t
                )
                if band == "similar":
                    fin2_similar.append(wid)
                if band == "match" and (
                    best_fin2 is None or sc > best_fin2[1]
                ):
                    best_fin2 = (wid, sc)
            status["similar_wine_ids"] = fin2_similar
            decision["similar_ids"] = fin2_similar
            decision["fin2_forced"] = not use_fin2
            if best_fin2 is not None:
                wid, conf_f = best_fin2
                _set_matched(
                    wid,
                    conf_f,
                    "xgb_dead_fin2",
                    {
                        "final_score2": conf_f,
                        "final_ocr": final_ocr_engine,
                        "cosine_similarity": cosine_by_id.get(wid),
                        "name": by_cos.get(wid, {}).get("name"),
                        "winery": by_cos.get(wid, {}).get("winery"),
                        "xgb_dead": True,
                        "fin2_forced": not use_fin2,
                        "max_xgb_score": decision.get("max_xgb_score"),
                    },
                )
                decision["band"] = "match"
                decision["fallback"] = "fin2"
                decision["match_ids"] = [wid]
                picked = True
            if not picked:
                best_id = None
                best_cos = -1.0
                for wid, cos in cosine_by_id.items():
                    if int(wid) in exclusive_reject_ids:
                        continue
                    if float(cos) > best_cos:
                        best_cos = float(cos)
                        best_id = int(wid)
                decision["cosine_best"] = (
                    {"id": best_id, "cosine": best_cos}
                    if best_id is not None
                    else None
                )
                if best_id is not None and best_cos >= float(empty_ocr_cos_thr):
                    _set_matched(
                        best_id,
                        best_cos,
                        "xgb_dead_cosine",
                        {
                            "cosine_similarity": best_cos,
                            "name": by_cos.get(best_id, {}).get("name"),
                            "winery": by_cos.get(best_id, {}).get("winery"),
                            "xgb_dead": True,
                            "max_xgb_score": decision.get("max_xgb_score"),
                        },
                    )
                    decision["band"] = "match"
                    decision["fallback"] = "cosine"
                    decision["match_ids"] = [best_id]
                else:
                    decision["band"] = "none"
                    decision["note"] = (
                        "xgb dead; Soft TF-IDF no match and max cosine "
                        "below empty_ocr_cosine_threshold"
                        if best_id is not None
                        else "xgb dead; no eligible candidates for fallback"
                    )
        else:
            match_ids: list[int] = []
            similar_ids: list[int] = []
            for wid, sc in scored_rows:
                band = score_band(sc, match=match_t, similar=similar_t)
                if band == "match":
                    match_ids.append(wid)
                elif band == "similar":
                    similar_ids.append(wid)
            status["similar_wine_ids"] = similar_ids
            decision["match_ids"] = match_ids
            decision["similar_ids"] = similar_ids

            # Pick best among match-band.
            # fin1/fin2: score already is FinalScore; xgb/crenc: pick by *_fin.
            best_pair: tuple[int, float] | None = None
            match_set = set(match_ids)
            if method in ("xgb", "crenc", "crenc_srv"):
                fbo = (status.get("steps") or {}).get("final_by_ocr") or {}
                primary = fbo.get("primary") if isinstance(fbo, dict) else None
                pack = (
                    ((fbo.get("by_ocr") or {}).get(str(primary)) or {})
                    if primary
                    else {}
                )
                fin_by = pack.get("by_id") if isinstance(pack, dict) else None
                if isinstance(fin_by, dict) and fin_by:
                    for wid in match_ids:
                        try:
                            fin_v = float(fin_by.get(str(wid)) or 0.0)
                        except (TypeError, ValueError):
                            continue
                        if best_pair is None or fin_v > best_pair[1]:
                            best_pair = (wid, fin_v)
                else:
                    step_key = "xgb_match" if method == "xgb" else "crenc_match"
                    fin_key = "xgb_fin" if method == "xgb" else "crenc_fin"
                    by_id = (
                        ((status.get("steps") or {}).get(step_key) or {}).get("by_id")
                        or {}
                    )
                    by_reuse = (
                        ((status.get("steps") or {}).get(step_key) or {}).get(
                            "by_reuse_sid"
                        )
                        or {}
                    )
                    for wid in match_ids:
                        score_row = by_id.get(str(wid)) or {}
                        try:
                            fin_v = float(score_row.get(fin_key) or 0.0)
                        except (TypeError, ValueError):
                            fin_v = 0.0
                        if isinstance(by_reuse, dict):
                            for rv in by_reuse.values():
                                if not isinstance(rv, dict):
                                    continue
                                try:
                                    if int(rv.get("id") or 0) != int(wid):
                                        continue
                                    fin_r = float(rv.get(fin_key) or 0.0)
                                except (TypeError, ValueError):
                                    continue
                                if fin_r > fin_v:
                                    fin_v = fin_r
                        if best_pair is None or fin_v > best_pair[1]:
                            best_pair = (wid, fin_v)

            else:
                for wid, sc in scored_rows:
                    if wid not in match_set:
                        continue
                    if best_pair is None or sc > best_pair[1]:
                        best_pair = (wid, sc)

            if best_pair is not None:
                wid, conf_f = best_pair
                if method == "fin1":
                    _set_matched(
                        wid,
                        conf_f,
                        "final_score",
                        {
                            "final_score": conf_f,
                            "final_ocr": final_ocr_engine,
                            "name": by_cos.get(wid, {}).get("name"),
                            "winery": by_cos.get(wid, {}).get("winery"),
                            "cosine_similarity": cosine_by_id.get(wid),
                        },
                    )
                elif method == "fin2":
                    _set_matched(
                        wid,
                        conf_f,
                        "final_score2",
                        {
                            "final_score2": conf_f,
                            "final_ocr": final_ocr_engine,
                            "name": by_cos.get(wid, {}).get("name"),
                            "winery": by_cos.get(wid, {}).get("winery"),
                            "cosine_similarity": cosine_by_id.get(wid),
                        },
                    )
                elif method == "xgb":
                    xgb_step_now = (status.get("steps") or {}).get("xgb_match") or {}
                    score_row = (xgb_step_now.get("by_id") or {}).get(str(wid)) or {}
                    by_reuse_now = xgb_step_now.get("by_reuse_sid") or {}
                    reuse_from_sid = None
                    if isinstance(by_reuse_now, dict):
                        for rv in by_reuse_now.values():
                            if not isinstance(rv, dict):
                                continue
                            try:
                                if int(rv.get("id") or 0) != int(wid):
                                    continue
                                if float(rv.get("xgb_fin") or 0) >= float(
                                    score_row.get("xgb_fin") or 0
                                ):
                                    score_row = rv
                                    reuse_from_sid = rv.get("search_photos_id")
                            except (TypeError, ValueError):
                                continue
                    extra_matched: dict[str, Any] = {
                        "xgb_score": score_row.get("xgb_score"),
                        "xgb_fin": conf_f,
                        "final_ocr": final_ocr_engine,
                        "cosine_similarity": score_row.get("cosine")
                        or cosine_by_id.get(wid),
                        "name": by_cos.get(wid, {}).get("name"),
                        "winery": by_cos.get(wid, {}).get("winery"),
                    }
                    if score_row.get("compare_source") == "previous_search_ocr":
                        extra_matched["from_previous_search"] = True
                        extra_matched["previous_search_photos_id"] = reuse_from_sid
                        extra_matched["xgb_compare_source"] = "previous_search_ocr"
                    _set_matched(wid, conf_f, "xgb_fin", extra_matched)
                elif method in ("crenc", "crenc_srv"):
                    score_row = (
                        ((status.get("steps") or {}).get("crenc_match") or {}).get("by_id")
                        or {}
                    ).get(str(wid)) or {}
                    _set_matched(
                        wid,
                        conf_f,
                        "crenc_fin",
                        {
                            "crenc_score": score_row.get("crenc_score"),
                            "crenc_fin": conf_f,
                            "final_ocr": final_ocr_engine,
                            "cosine_similarity": score_row.get("cosine")
                            or cosine_by_id.get(wid),
                            "name": by_cos.get(wid, {}).get("name"),
                            "winery": by_cos.get(wid, {}).get("winery"),
                        },
                    )
                elif method == "openai_txt":
                    score_row = (
                        ((status.get("steps") or {}).get("openai_txt_match") or {}).get(
                            "by_id"
                        )
                        or {}
                    ).get(str(wid)) or {}
                    _set_matched(
                        wid,
                        conf_f,
                        "llm_txt",
                        {
                            "llm_txt": conf_f,
                            "final_ocr": final_ocr_engine,
                            "cosine_similarity": cosine_by_id.get(wid),
                            "name": by_cos.get(wid, {}).get("name"),
                            "winery": by_cos.get(wid, {}).get("winery"),
                        },
                    )
                decision["band"] = "match"
            elif similar_ids:
                decision["band"] = "similar"
                decision["note"] = "scores in similar band only — final empty"
            else:
                decision["band"] = "none"
    else:
        decision["band"] = "none"
        decision["note"] = "final_score_method not in text_match_methods"

    status["steps"]["text_match_decision"] = decision

    _flush_hf_start_status()
    _time_total()
    # Аналоги / похожие — для сайта, вне общего timeline (после total).
    # Match: критерии из каталожного winner → «похожие вина».
    # Нет match: критерии OCR искомого (как раньше).
    # eval_mode=1: аналоги не считаем (тот же match, что eval=0).
    if not eval_flag:
        try:
            matched_id = status.get("matched_wine_id")
            if matched_id is not None:
                status["analogs"] = find_analogs_from_wine(
                    int(matched_id),
                    cosine_by_id=cosine_by_id,
                )
            else:
                status["analogs"] = find_analogs(
                    str(xgb_ocr_text or exclusive_query or ""),
                    cosine_by_id=cosine_by_id,
                )
        except Exception as exc:  # noqa: BLE001
            status["analogs"] = {"ok": False, "error": str(exc), "items": []}
        _plog(
            "OK analogs",
            ms=status["analogs"].get("ms"),
            n=len(status["analogs"].get("items") or []),
            source=status["analogs"].get("source"),
            reason=status["analogs"].get("reason"),
        )
    _plog(
        "DONE pipeline done",
        ok=status.get("ok"),
        ms=status["timings_ms"].get("total"),
        matched=status.get("matched_wine_id"),
        conf=status.get("matched_wine_confidence"),
        similar_n=len(status.get("similar_wine_ids") or []),
        error=status.get("error"),
    )
    _save_status(db, row, status, commit=True)
    return status
