from pathlib import Path
import asyncio
import mimetypes
import re
from io import BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.db.config import SEARCH_PHOTO_DIR
from app.db.models import SearchPhoto, Wine, WineSitemap
from app.media import label_url, photo_url
from app.pipeline.findwine import run_findwine
from app.pipeline.image_io import open_image_rgb
from app.pipeline.runtime_settings import (
    get_settings,
    save_settings,
    settings_public_view,
)
from app.pipeline.version import ALGORITHM_UPDATED_AT, ALGORITHM_VERSION
from app.scan_history_denorm import (
    _is_official_match_source,
    apply_status_eval_flags,
    eval_flags_for_manual_match,
    history_resolve_matched_wine_id as _history_resolve_matched_wine_id,
    sync_search_photo_history_columns,
)
from app.schemas import (
    FilterFacet,
    FiltersResponse,
    EvalPredictResponse,
    FindWineCandidate,
    FindWineResponse,
    ManualWineOut,
    ManualWineUpdate,
    PipelineSettingsOut,
    PipelineSettingsUpdate,
    ScanHistoryEvalUpdate,
    ScanHistoryItem,
    ScanHistoryReportCounts,
    ScanHistoryReportErrorItem,
    ScanHistoryReportMetrics,
    ScanHistoryReportResponse,
    ScanHistoryReportScoreDist,
    ScanHistoryReportTimingBin,
    ScanHistoryReportTimingDist,
    ScanHistoryResponse,
    ScanHistoryXgbTopItem,
    ScanResponse,
    WineListResponse,
    WineOut,
)

router = APIRouter(prefix="/api")
eval_router = APIRouter()

FILTER_FIELDS = ("category", "color", "region", "grape_variety", "winery", "name")


def _as_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        for part in raw.split(","):
            item = part.strip()
            if item:
                out.append(item)
    return out


def _word_start_regex(q: str) -> str | None:
    """Match query only at the start of a word (letters/digits)."""
    term = q.strip()
    if not term:
        return None
    escaped = re.escape(term)
    # Start of string or non-alphanumeric (Latin + Cyrillic) boundary
    return rf"(^|[^0-9A-Za-zА-Яа-яЁё]){escaped}"


def _apply_filters(
    stmt: Select,
    *,
    category: list[str],
    color: list[str],
    region: list[str],
    grape_variety: list[str],
    winery: list[str],
    name: list[str],
    q: str | None,
) -> Select:
    if category:
        stmt = stmt.where(Wine.category.in_(category))
    if color:
        stmt = stmt.where(Wine.color.in_(color))
    if region:
        stmt = stmt.where(Wine.region.in_(region))
    if grape_variety:
        stmt = stmt.where(Wine.grape_variety.in_(grape_variety))
    if winery:
        stmt = stmt.where(Wine.winery.in_(winery))
    if name:
        stmt = stmt.where(Wine.name.in_(name))
    if q:
        rx = _word_start_regex(q)
        if rx:
            stmt = stmt.where(
                or_(
                    Wine.name.op("~*")(rx),
                    Wine.winery.op("~*")(rx),
                    Wine.category.op("~*")(rx),
                    Wine.grape_variety.op("~*")(rx),
                    Wine.color.op("~*")(rx),
                )
            )
    return stmt


def _sitemap_image_map(db: Session, slugs: list[str]) -> dict[str, str]:
    clean = [s for s in slugs if s]
    if not clean:
        return {}
    rows = db.execute(
        select(WineSitemap.slug, WineSitemap.image_file).where(
            WineSitemap.slug.in_(clean),
            WineSitemap.image_file.is_not(None),
        )
    ).all()
    return {str(slug): str(image_file) for slug, image_file in rows if image_file}


def _serialize(
    wine: Wine,
    *,
    sitemap_image_file: str | None = None,
    include_label_ocr: bool = False,
) -> WineOut:
    return WineOut(
        id=wine.id,
        wineries_id=wine.wineries_id,
        name=wine.name,
        category=wine.category,
        color=wine.color,
        region=wine.region,
        grape_variety=wine.grape_variety,
        description=wine.description,
        winery=wine.winery,
        slug=wine.slug,
        photo_name=wine.photo_name,
        photo_url=photo_url(
            wine.photo_name,
            slug=wine.slug,
            sitemap_image_file=sitemap_image_file,
        ),
        crop=wine.crop,
        label=wine.label,
        label_url=label_url(wine.photo_name, slug=wine.slug),
        label_ocr=wine.label_ocr if include_label_ocr else None,
    )


@router.get("/wines", response_model=WineListResponse)
def list_wines(
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=100),
    category: list[str] | None = Query(None),
    color: list[str] | None = Query(None),
    region: list[str] | None = Query(None),
    grape_variety: list[str] | None = Query(None),
    winery: list[str] | None = Query(None),
    name: list[str] | None = Query(None),
    q: str | None = Query(None),
    db: Session = Depends(get_db),
) -> WineListResponse:
    filters = dict(
        category=_as_list(category),
        color=_as_list(color),
        region=_as_list(region),
        grape_variety=_as_list(grape_variety),
        winery=_as_list(winery),
        name=_as_list(name),
        q=q,
    )
    base = select(Wine)
    base = _apply_filters(base, **filters)
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = db.scalars(
        base.order_by(Wine.name.asc(), Wine.id.asc()).offset(offset).limit(limit)
    ).all()
    sitemap_map = _sitemap_image_map(db, [w.slug for w in rows if w.slug])
    items = [
        _serialize(w, sitemap_image_file=sitemap_map.get(w.slug or ""))
        for w in rows
    ]
    return WineListResponse(
        items=items,
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + len(items) < total,
    )


@router.get("/wines/filters", response_model=FiltersResponse)
def wine_filters(db: Session = Depends(get_db)) -> FiltersResponse:
    def facets(column) -> list[FilterFacet]:
        rows = db.execute(
            select(column, func.count())
            .where(column.is_not(None))
            .where(column != "")
            .group_by(column)
            .order_by(func.count().desc(), column.asc())
        ).all()
        return [FilterFacet(value=str(value), count=int(count)) for value, count in rows]

    return FiltersResponse(
        category=facets(Wine.category),
        color=facets(Wine.color),
        region=facets(Wine.region),
        grape_variety=facets(Wine.grape_variety),
        winery=facets(Wine.winery),
        name=facets(Wine.name),
    )


@router.get("/wines/{slug}", response_model=WineOut)
def get_wine(slug: str, db: Session = Depends(get_db)) -> WineOut:
    wine = db.scalar(select(Wine).where(Wine.slug == slug))
    if wine is None:
        raise HTTPException(status_code=404, detail="Wine not found")
    sitemap_map = _sitemap_image_map(db, [slug])
    return _serialize(
        wine,
        sitemap_image_file=sitemap_map.get(slug),
        include_label_ocr=True,
    )


@router.post("/scan", response_model=ScanResponse)
async def scan_wine(file: UploadFile = File(...)) -> ScanResponse:
    data = await file.read()
    return ScanResponse(
        filename=file.filename or "upload",
        message="Файл получен. Распознавание этикетки будет подключено на следующем шаге.",
        size=len(data),
    )


@router.post("/preview-image")
async def preview_image(file: UploadFile = File(...)) -> Response:
    """Decode any supported raster (HEIC/WebP/…) → JPEG for browser preview."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Пустой файл")
    try:
        img, _ = open_image_rgb(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Не удалось прочитать изображение: {exc}"
        ) from exc

    from PIL import Image as PilImage

    # Cap preview size for UI speed
    max_side = 1600
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            resample=PilImage.Resampling.LANCZOS,
        )

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


def _cand_list_from_status(status: dict, key: str) -> list[FindWineCandidate]:
    cand = (status.get("steps") or {}).get("candidates") or {}
    items = (cand.get(key) or {}).get("items") or []
    out: list[FindWineCandidate] = []
    for it in items:
        try:
            out.append(FindWineCandidate.model_validate(it))
        except Exception:  # noqa: BLE001
            continue
    return out


def _findwine_response_from_status(
    status: dict,
    *,
    manual_wines_id: int | None = None,
) -> FindWineResponse:
    from app.pipeline.findwine import _json_safe

    status = _json_safe(status) if isinstance(status, dict) else {}
    crops_filename = status.get("crops_filename")
    label_filename = status.get("label_filename")
    matched = status.get("matched_wine") or {}
    matched_id = status.get("matched_wine_id")
    conf = status.get("matched_wine_confidence")
    try:
        conf_f = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf_f = None
    mid_manual = manual_wines_id
    if mid_manual is None and status.get("manual_wines_id") is not None:
        try:
            mid_manual = int(status["manual_wines_id"])
        except (TypeError, ValueError):
            mid_manual = None
    return FindWineResponse(
        search_photos_id=status.get("search_photos_id"),
        crops_filename=crops_filename,
        label_filename=label_filename,
        crops_url=(
            f"/api/search-photos/{crops_filename}" if crops_filename else None
        ),
        label_url=(
            f"/api/search-photos/{label_filename}" if label_filename else None
        ),
        algorithm_version=status.get("algorithm_version") or ALGORITHM_VERSION,
        algorithm_updated_at=status.get("algorithm_updated_at")
        or ALGORITHM_UPDATED_AT,
        candidates_siglip2=_cand_list_from_status(status, "siglip2"),
        candidates_dinov3=_cand_list_from_status(status, "dinov3"),
        status=status,
        matched_wine_id=int(matched_id) if matched_id is not None else None,
        matched_wine_confidence=conf_f,
        matched_wine_name=matched.get("name"),
        matched_wine_slug=matched.get("slug"),
        matched_wine_photo_url=matched.get("photo_url"),
        manual_wines_id=mid_manual,
    )


def _winner_slug(result: FindWineResponse, db: Session) -> str | None:
    """Slug победителя findwine: карточка matched_wine, иначе wines.slug по id."""
    slug = (result.matched_wine_slug or "").strip()
    if slug:
        return slug
    matched = result.status.get("matched_wine") if isinstance(result.status, dict) else None
    if isinstance(matched, dict):
        raw = matched.get("slug")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    wid = result.matched_wine_id
    if wid is None and isinstance(result.status, dict):
        raw_id = result.status.get("matched_wine_id")
        try:
            wid = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            wid = None
    if wid is None:
        return None
    wine = db.get(Wine, int(wid))
    if wine is not None and wine.slug:
        return str(wine.slug).strip() or None
    return None


@router.post("/findwine", response_model=FindWineResponse)
async def findwine(
    file: UploadFile = File(...),
    eval: int = Query(0, ge=0, le=1),
) -> FindWineResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Пустой файл")
    filename = file.filename or "upload.bin"
    eval_mode = int(eval)

    def _job() -> tuple[dict, int | None]:
        # Own session: run_findwine is sync/CPU-bound; keep event loop free
        # so GET /findwine/{id} (saved scans) is not blocked.
        db = SessionLocal()
        try:
            status = run_findwine(
                db,
                data=data,
                filename=filename,
                eval_mode=eval_mode,
            )
            manual_id = None
            sp_id = status.get("search_photos_id")
            if sp_id is not None:
                row = db.get(SearchPhoto, int(sp_id))
                if row is not None:
                    manual_id = row.manual_wines_id
            return status, manual_id
        finally:
            db.close()

    status, manual_id = await asyncio.to_thread(_job)
    return _findwine_response_from_status(status, manual_wines_id=manual_id)


@eval_router.post("/v1/eval/predict", response_model=EvalPredictResponse)
async def eval_predict(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> EvalPredictResponse:
    """Контракт participant_test.sh: multipart-поле image → {\"slug\": \"...\"}."""
    result = await findwine(file=image, eval=1)
    return EvalPredictResponse(slug=_winner_slug(result, db))


def _browser_image_filename(*candidates: str | None) -> str | None:
    """Pick first existing-looking name browsers can render (not HEIC/AVIF raw)."""
    browser_ext = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    fallback: str | None = None
    for name in candidates:
        if not name or not isinstance(name, str):
            continue
        safe = Path(name).name
        if not safe:
            continue
        if fallback is None:
            fallback = safe
        if Path(safe).suffix.lower() in browser_ext:
            return safe
    return fallback


def _best_query_ocr_text(
    status: dict,
    ocr_texts: dict[str, str | None],
) -> str | None:
    """Prefer primary OCR text / best channel for history OCR column."""
    steps = status.get("steps") or {}
    ocr = steps.get("ocr") or {}
    for key in (
        "text",
        "text_aggregated",
    ):
        t = str(ocr.get(key) or "").strip()
        if t:
            return t
    for eng in (
        "google_vision",
        "yandex",
        "gemini",
        "openai",
        "deepseek",
        "qwen",
    ):
        step = steps.get(f"ocr_{eng}") or {}
        t = str(step.get("text") or "").strip()
        if t:
            return t
        t = str((ocr_texts.get(eng) or "")).strip()
        if t:
            return t
    help_ = steps.get("ocr_wine_id") or {}
    bf = help_.get("best_final") or {}
    from_ch = bf.get("text_from")
    if from_ch and ocr_texts.get(str(from_ch)):
        t = str(ocr_texts.get(str(from_ch)) or "").strip()
        if t:
            return t
    for _k, v in ocr_texts.items():
        t = str(v or "").strip()
        if t:
            return t
    return None


def _history_scores(
    status: dict,
    *,
    matched_id: int | None,
    confidence: float | None,
) -> dict[str, float | str | None]:
    """Scores mirroring candidate cards (0..1 scale where applicable).

    Only metrics that were actually computed for this scan are returned
    (based on status.settings.text_match_methods + present step values).
    """
    steps = status.get("steps") or {}
    settings = status.get("settings") or {}
    methods = {
        str(m).strip().lower()
        for m in (settings.get("text_match_methods") or [])
        if str(m).strip()
    }
    # Legacy scans without settings: infer from present steps / scores.
    infer = not bool(methods)
    use_fin1 = infer or "fin1" in methods
    use_fin2 = infer or "fin2" in methods
    use_xgb = infer or "xgb" in methods
    use_crenc = infer or ("crenc" in methods) or ("crenc_srv" in methods)

    help_ = steps.get("ocr_wine_id") or {}
    ranked = help_.get("final_ranked") or []
    ranked2 = help_.get("final_ranked2") or []
    row: dict = {}
    row2: dict = {}
    if matched_id is not None:
        for r in ranked:
            if isinstance(r, dict) and int(r.get("id") or 0) == int(matched_id):
                row = r
                break
        for r in ranked2:
            if isinstance(r, dict) and int(r.get("id") or 0) == int(matched_id):
                row2 = r
                break
    # No match → do not attribute ocr_wine_id best_* scores to the score column
    # (history shows xgb_top cards instead).
    if matched_id is not None and not row:
        row = help_.get("best_final") or {}
        if not isinstance(row, dict):
            row = {}
        try:
            if int(row.get("id") or 0) != int(matched_id):
                row = {}
        except (TypeError, ValueError):
            row = {}
    if matched_id is not None and not row2:
        bf2 = help_.get("best_final2") or {}
        if isinstance(bf2, dict) and int(bf2.get("id") or 0) == int(matched_id):
            row2 = bf2
    matched = status.get("matched_wine") or {}
    matched_source = str(
        matched.get("source") or status.get("matched_wine_source") or ""
    ).lower()

    def _f(v: object) -> float | None:
        try:
            if v is None:
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    cos = _f(row.get("cosine"))
    if cos is None:
        cos = _f(row2.get("cosine"))
    if cos is None:
        cos = _f(matched.get("cosine_similarity"))
    if cos is None and matched_id is not None:
        cos_map = status.get("candidates_cosine") or {}
        cos = _f(cos_map.get(str(matched_id)) or cos_map.get(matched_id))

    txt1 = fin1 = None
    if use_fin1:
        txt1 = _f(row.get("score_01"))
        if txt1 is None:
            sc = _f(row.get("score"))
            if sc is not None:
                txt1 = sc / 100.0 if sc > 1.5 else sc
        if txt1 is None:
            txt1 = _f(matched.get("text_score_01"))
        if txt1 is None:
            sc = _f(matched.get("text_score"))
            if sc is not None:
                txt1 = sc / 100.0 if sc > 1.5 else sc

        fin1 = _f(row.get("final_score"))
        if fin1 is None:
            fin1 = _f(matched.get("final_score"))
        # confidence → fin1 только если победитель именно по fin1
        if fin1 is None and matched_source in {"final_score", "fin1", ""}:
            fin1 = _f(confidence)

    txt2 = fin2 = None
    if use_fin2:
        txt2 = _f(row.get("score2_01"))
        if txt2 is None:
            txt2 = _f(row2.get("score2_01"))
        if txt2 is None:
            sc2 = _f(row.get("score2"))
            if sc2 is None:
                sc2 = _f(row2.get("score2"))
            if sc2 is not None:
                txt2 = sc2 / 100.0 if sc2 > 1.5 else sc2
        fin2 = _f(row.get("final_score2"))
        if fin2 is None:
            fin2 = _f(row2.get("final_score2"))
        if fin2 is None and matched_source in {"final_score2", "fin2", "soft_tfidf"}:
            fin2 = _f(confidence)

    xgb = xgb_fin = None
    if use_xgb:
        xgb_step = steps.get("xgb_match") or {}
        xgb_row: dict = {}
        if matched_id is not None:
            by_id = xgb_step.get("by_id") or {}
            if isinstance(by_id, dict):
                xgb_row = by_id.get(str(int(matched_id))) or by_id.get(matched_id) or {}
                if not isinstance(xgb_row, dict):
                    xgb_row = {}
        # No best_xgb_fin fallback without an official match.
        xgb = _f(xgb_row.get("xgb_score"))
        xgb_fin = _f(xgb_row.get("xgb_fin"))
        if xgb_fin is None and matched_source in {"xgb", "xgb_fin"}:
            xgb_fin = _f(confidence)

    crenc = crenc_fin = None
    if use_crenc:
        crenc_step = steps.get("crenc_match") or {}
        crenc_row: dict = {}
        if matched_id is not None:
            by_id = crenc_step.get("by_id") or {}
            if isinstance(by_id, dict):
                crenc_row = (
                    by_id.get(str(int(matched_id))) or by_id.get(matched_id) or {}
                )
                if not isinstance(crenc_row, dict):
                    crenc_row = {}
        if not crenc_row:
            bf = crenc_step.get("best_crenc_fin")
            if isinstance(bf, dict):
                crenc_row = bf
        crenc = _f(crenc_row.get("crenc_score") or crenc_row.get("score"))
        crenc_fin = _f(crenc_row.get("crenc_fin"))
        if crenc_fin is None and matched_source in {
            "crenc",
            "crenc_fin",
            "crenc_srv",
        }:
            crenc_fin = _f(confidence)

    out: dict[str, float | str | None] = {
        "cos": round(cos, 4) if cos is not None else None,
        "txt1": round(txt1, 4) if txt1 is not None else None,
        "fin1": round(fin1, 4) if fin1 is not None else None,
        "txt2": round(txt2, 4) if txt2 is not None else None,
        "fin2": round(fin2, 4) if fin2 is not None else None,
        "xgb": round(xgb, 4) if xgb is not None else None,
        "xgb_fin": round(xgb_fin, 4) if xgb_fin is not None else None,
        "crenc": round(crenc, 4) if crenc is not None else None,
        "crenc_fin": round(crenc_fin, 4) if crenc_fin is not None else None,
        "text_from": row.get("text_from")
        or row2.get("text_from")
        or matched.get("text_from"),
    }
    # Per-channel FinalScore from visual_support of best variant
    per = help_.get("per_variant") or {}
    channel_bits: list[str] = []
    for ch in (
        "google_vision",
        "yandex",
        "gemini",
        "openai",
        "deepseek",
        "qwen",
    ):
        entry = per.get(ch) or {}
        best = entry.get("best_final") or {}
        if matched_id is not None:
            for vs in entry.get("visual_support") or []:
                if isinstance(vs, dict) and int(vs.get("id") or 0) == int(matched_id):
                    best = vs
                    break
        fs = _f(best.get("final_score"))
        # 0.0 = канал не считал FinalScore (fin1 выкл) — не показываем Score V/Y/…
        if fs is None or abs(float(fs)) < 1e-12:
            continue
        letter = {
            "google_vision": "V",
            "yandex": "Y",
            "gemini": "G",
            "openai": "O",
            "deepseek": "K",
            "qwen": "Q",
        }.get(ch, ch[:1].upper())
        out[f"score_{letter}"] = round(fs, 4)
        channel_bits.append(f"Score {letter} {fs:.2f}")
    if channel_bits:
        out["channels_line"] = " · ".join(channel_bits)
    return out


def _history_xgb_top_raw(status: dict, *, limit: int = 7) -> list[dict]:
    """Raw top-N candidates when search found no final match.

    Prefer XGB_fin ranking; else Soft TF-IDF fin2; else cosine from
    SigLIP2/DINOv3 candidate lists in status.
    """
    lim = max(1, min(int(limit), 20))

    def _f(v: object) -> float:
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return -1.0

    xgb_step = (status.get("steps") or {}).get("xgb_match") or {}
    by_id = xgb_step.get("by_id") or {}
    if isinstance(by_id, dict) and by_id:
        ranked: list[dict] = []
        for wid_s, raw in by_id.items():
            if not isinstance(raw, dict):
                continue
            try:
                wid = int(raw.get("id") if raw.get("id") is not None else wid_s)
            except (TypeError, ValueError):
                continue
            ranked.append(
                {
                    "id": wid,
                    "xgb_fin": _f(raw.get("xgb_fin")),
                    "xgb_score": _f(raw.get("xgb_score")),
                    "cosine": _f(raw.get("cosine")),
                    "fin2": _f(raw.get("final_score2")),
                }
            )
        ranked.sort(
            key=lambda t: (
                -t["xgb_fin"],
                -t["xgb_score"],
                -t["cosine"],
                t["id"],
            )
        )
        # Soft TF-IDF scores for the same wines (dead-XGB fallback display)
        help_fin = (status.get("steps") or {}).get("ocr_wine_id") or {}
        fin2_map: dict[int, float] = {}
        for raw in help_fin.get("final_ranked2") or []:
            if not isinstance(raw, dict) or raw.get("id") is None:
                continue
            try:
                fw = int(raw["id"])
                fv = _f(raw.get("final_score2"))
            except (TypeError, ValueError):
                continue
            if fv >= 0:
                fin2_map[fw] = fv
        if not fin2_map:
            for entry in (help_fin.get("per_variant") or {}).values():
                if not isinstance(entry, dict):
                    continue
                for row in entry.get("visual_support") or []:
                    if not isinstance(row, dict) or row.get("id") is None:
                        continue
                    try:
                        fw = int(row["id"])
                        fv = _f(row.get("final_score2"))
                    except (TypeError, ValueError):
                        continue
                    if fv >= 0 and (fw not in fin2_map or fv > fin2_map[fw]):
                        fin2_map[fw] = fv
        for t in ranked:
            if t["fin2"] < 0 and t["id"] in fin2_map:
                t["fin2"] = fin2_map[t["id"]]
        return ranked[:lim]

    # Soft TF-IDF (fin2) from ocr_wine_id — used on dead-XGB fallback
    help_ = (status.get("steps") or {}).get("ocr_wine_id") or {}
    ranked2 = help_.get("final_ranked2") or []
    if isinstance(ranked2, list) and ranked2:
        out2: list[dict] = []
        for raw in ranked2:
            if not isinstance(raw, dict) or raw.get("id") is None:
                continue
            try:
                wid = int(raw["id"])
            except (TypeError, ValueError):
                continue
            out2.append(
                {
                    "id": wid,
                    "xgb_fin": -1.0,
                    "xgb_score": -1.0,
                    "cosine": _f(raw.get("cosine")),
                    "fin2": _f(raw.get("final_score2")),
                }
            )
        out2.sort(key=lambda t: (-t["fin2"], -t["cosine"], t["id"]))
        if out2:
            return out2[:lim]

    # Cosine from embedding candidate lists
    cand = (status.get("steps") or {}).get("candidates") or {}
    by_cos: dict[int, float] = {}
    for key in ("siglip2", "dinov3"):
        items = (cand.get(key) or {}).get("items") or []
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict) or it.get("id") is None:
                continue
            try:
                wid = int(it["id"])
            except (TypeError, ValueError):
                continue
            cos = _f(it.get("cosine_similarity"))
            prev = by_cos.get(wid)
            if prev is None or cos > prev:
                by_cos[wid] = cos
    if by_cos:
        ranked_cos = [
            {
                "id": wid,
                "xgb_fin": -1.0,
                "xgb_score": -1.0,
                "cosine": cos,
                "fin2": -1.0,
            }
            for wid, cos in by_cos.items()
        ]
        ranked_cos.sort(key=lambda t: (-t["cosine"], t["id"]))
        return ranked_cos[:lim]
    return []


def _history_build_xgb_top(
    raw_top: list[dict],
    wines_by_id: dict[int, Wine],
    sitemap_map: dict[str, str],
) -> list[ScanHistoryXgbTopItem]:
    out: list[ScanHistoryXgbTopItem] = []
    for t in raw_top:
        wid = int(t["id"])
        wine = wines_by_id.get(wid)
        slug = wine.slug if wine is not None else None
        sitemap_file = sitemap_map.get(str(slug)) if slug else None
        xgb_fin = t.get("xgb_fin")
        xgb_score = t.get("xgb_score")
        cosine = t.get("cosine")
        out.append(
            ScanHistoryXgbTopItem(
                id=wid,
                name=wine.name if wine is not None else None,
                slug=slug,
                winery=wine.winery if wine is not None else None,
                label=wine.label if wine is not None else None,
                label_url=(
                    label_url(wine.photo_name, slug=wine.slug)
                    if wine is not None
                    else None
                ),
                photo_url=(
                    photo_url(
                        wine.photo_name,
                        slug=wine.slug,
                        sitemap_image_file=sitemap_file,
                    )
                    if wine is not None
                    else None
                ),
                xgb_score=round(xgb_score, 4) if xgb_score is not None and xgb_score >= 0 else None,
                xgb_fin=round(xgb_fin, 4) if xgb_fin is not None and xgb_fin >= 0 else None,
                cosine=round(cosine, 4) if cosine is not None and cosine >= 0 else None,
                fin2=(
                    round(float(t["fin2"]), 4)
                    if t.get("fin2") is not None and float(t.get("fin2")) >= 0
                    else None
                ),
            )
        )
    return out


def _history_item_from_row(
    row: SearchPhoto,
    wine: Wine | None = None,
    *,
    xgb_top: list[ScanHistoryXgbTopItem] | None = None,
) -> ScanHistoryItem:
    status = dict(row.status or {})
    steps = status.get("steps") or {}
    save = steps.get("save") or {}
    normalize = steps.get("normalize") or {}
    ocr = steps.get("ocr") or {}
    variants = ocr.get("variants") or {}
    # Prefer engine×preprocess keys; fall back to legacy A–D
    ocr_texts: dict[str, str | None] = {}
    engine_keys = ocr.get("engines") or ["rapid", "easy", "surya", "tess"]
    for prep in ("A", "B", "C", "D"):
        for eng in engine_keys:
            key = f"{prep}_{eng}"
            if key in variants:
                ocr_texts[key] = ((variants.get(key) or {}).get("text") or None)
        if prep in variants and prep not in ocr_texts:
            ocr_texts[prep] = ((variants.get(prep) or {}).get("text") or None)
    for eng in ("gemini", "openai", "deepseek", "qwen", "yandex", "google_vision"):
        if eng in variants:
            ocr_texts[eng] = ((variants.get(eng) or {}).get("text") or None)
    if not ocr_texts:
        ocr_texts = {
            key: ((variants.get(key) or {}).get("text") or None)
            for key in ("A", "B", "C", "D")
        }

    # Искомая этикетка: предпочтительно label-crop
    query_filename = _browser_image_filename(
        status.get("label_filename"),
        status.get("crops_filename"),
        status.get("work_filename"),
        normalize.get("work_filename"),
        save.get("filename"),
    )
    query_photo_url = (
        f"/api/search-photos/{query_filename}" if query_filename else None
    )

    matched_raw = status.get("matched_wine") or {}
    # Always resolve from status — hist_* can be stale (pre-fix denorm).
    matched_id = _history_resolve_matched_wine_id(status)
    source = str(matched_raw.get("source") or "").strip().lower()
    # Use stored card when it is the official match we resolved (fin/xgb/crenc/…)
    matched: dict = {}
    if matched_id is not None and _is_official_match_source(source):
        try:
            if int(matched_raw.get("id") or 0) == int(matched_id):
                matched = matched_raw
        except (TypeError, ValueError):
            matched = {}
    elif matched_id is not None and isinstance(matched_raw, dict):
        try:
            if int(matched_raw.get("id") or 0) == int(matched_id):
                matched = matched_raw
        except (TypeError, ValueError):
            matched = {}

    confidence = None
    if (
        matched_id is not None
        and row.hist_wine_id is not None
        and int(row.hist_wine_id) == int(matched_id)
    ):
        confidence = row.hist_wine_confidence
    if confidence is None and matched:
        confidence = status.get("matched_wine_confidence")
        if confidence is None:
            confidence = matched.get("confidence") or matched.get("final_score")
    # Do not invent confidence from ocr_wine_id best_final when there is no match.

    timings = status.get("timings_ms") or {}
    total_ms = row.hist_total_ms
    if total_ms is None:
        total_ms = timings.get("total")
    try:
        total_ms_f = float(total_ms) if total_ms is not None else None
    except (TypeError, ValueError):
        total_ms_f = None
    try:
        conf_f = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        conf_f = None

    mid = int(matched_id) if matched_id is not None else None
    matched_label = matched.get("label")
    matched_photo = matched.get("label_url") or matched.get("photo_url")
    matched_name = matched.get("name")
    matched_slug = matched.get("slug")
    if (
        mid is not None
        and row.hist_wine_id is not None
        and int(row.hist_wine_id) == mid
        and row.hist_wine_slug
    ):
        matched_slug = matched_slug or row.hist_wine_slug
    if wine is not None:
        if not matched_label:
            matched_label = wine.label
        if not matched_name:
            matched_name = wine.name
        if not matched_slug:
            matched_slug = wine.slug
        if not matched_photo:
            matched_photo = label_url(wine.photo_name, wine.slug) or photo_url(
                wine.photo_name, wine.slug
            )

    created = row.created_at.isoformat() if row.created_at else None
    fp = int(row.hist_fp or 0)
    fn = int(row.hist_fn or 0)
    return ScanHistoryItem(
        id=row.id,
        created_at=created,
        query_photo_url=query_photo_url,
        query_filename=query_filename,
        total_ms=total_ms_f,
        ocr_texts=ocr_texts,
        query_ocr_text=_best_query_ocr_text(status, ocr_texts),
        matched_wine_id=mid,
        matched_wine_confidence=conf_f,
        matched_wine_name=matched_name,
        matched_wine_slug=matched_slug,
        matched_wine_photo_url=matched_photo,
        matched_wine_label=matched_label,
        xgb_top=xgb_top if mid is None else None,
        manual_wines_id=(
            int(row.manual_wines_id) if row.manual_wines_id is not None else None
        ),
        scores=_history_scores(status, matched_id=mid, confidence=conf_f),
        algorithm_version=status.get("algorithm_version"),
        false_positive=fp,
        false_negative=fn,
    )


@router.patch("/scan-history/{scan_id}/eval", response_model=ScanHistoryItem)
def update_scan_history_eval(
    scan_id: int,
    body: ScanHistoryEvalUpdate,
    db: Session = Depends(get_db),
) -> ScanHistoryItem:
    """Save mutual-exclusive FP/FN flags into search_photos.status.eval."""
    row = db.get(SearchPhoto, scan_id)
    if row is None:
        raise HTTPException(status_code=404, detail="scan not found")
    fp = 1 if int(body.false_positive or 0) else 0
    fn = 1 if int(body.false_negative or 0) else 0
    if fp and fn:
        raise HTTPException(
            status_code=400,
            detail="false_positive and false_negative cannot both be 1",
        )
    status = dict(row.status or {})
    status["eval"] = {"false_positive": fp, "false_negative": fn}
    row.status = status
    from sqlalchemy.orm.attributes import flag_modified

    flag_modified(row, "status")
    sync_search_photo_history_columns(row, status)
    db.add(row)
    db.commit()
    db.refresh(row)
    st = dict(row.status or {})
    mid = row.hist_wine_id
    if mid is None:
        mid = _history_resolve_matched_wine_id(st)
    wine = db.get(Wine, mid) if mid is not None else None
    xgb_top: list[ScanHistoryXgbTopItem] | None = None
    if mid is None:
        raw_top = _history_xgb_top_raw(st)
        if raw_top:
            top_ids = [int(t["id"]) for t in raw_top]
            wines_map: dict[int, Wine] = {}
            for w in db.scalars(select(Wine).where(Wine.id.in_(top_ids))).all():
                wines_map[int(w.id)] = w
            sitemap_map = _sitemap_image_map(
                db, [w.slug for w in wines_map.values() if w.slug]
            )
            xgb_top = _history_build_xgb_top(raw_top, wines_map, sitemap_map)
    return _history_item_from_row(row, wine, xgb_top=xgb_top)


def _safe_div(a: float, b: float) -> float | None:
    if b <= 0:
        return None
    return float(a) / float(b)


def _classify_scan_eval(
    *,
    matched_id: int | None,
    manual_id: int | None,
    hist_fp: int,
    hist_fn: int,
) -> tuple[str, str]:
    """Класс TP|FP|TN|FN и причина.

    Ручной manual_wines_id важнее чекбоксов:
    - есть финальное вино и другое manual → FP;
    - нет победителя, но есть manual → FN.
    """
    mid = int(matched_id) if matched_id is not None else None
    man = int(manual_id) if manual_id is not None else None
    fp = 1 if int(hist_fp or 0) else 0
    fn = 1 if int(hist_fn or 0) else 0
    if mid is not None and man is not None and mid != man:
        return "FP", "manual_differs_from_matched"
    if mid is None and man is not None:
        return "FN", "manual_without_matched"
    if fp:
        return "FP", "checkbox_fp"
    if fn:
        return "FN", "checkbox_fn"
    if mid is not None:
        return "TP", "matched_ok"
    return "TN", "no_match_ok"


def _report_candidate_pool(status: dict) -> dict[int, dict[str, float]]:
    """wine_id → {cosine, xgb_score} из xgb_match.by_id или embedding-кандидатов."""
    out: dict[int, dict[str, float]] = {}
    xgb = ((status.get("steps") or {}).get("xgb_match") or {}).get("by_id") or {}
    if isinstance(xgb, dict):
        for k, v in xgb.items():
            if not isinstance(v, dict):
                continue
            try:
                wid = int(k)
            except (TypeError, ValueError):
                continue
            row: dict[str, float] = {}
            if v.get("cosine") is not None:
                try:
                    row["cosine"] = float(v["cosine"])
                except (TypeError, ValueError):
                    pass
            if v.get("xgb_score") is not None:
                try:
                    row["xgb_score"] = float(v["xgb_score"])
                except (TypeError, ValueError):
                    pass
            if row:
                out[wid] = row
    if out:
        return out
    for key in ("candidates_siglip2", "candidates_dinov3"):
        for c in status.get(key) or []:
            if not isinstance(c, dict) or c.get("id") is None:
                continue
            try:
                wid = int(c["id"])
            except (TypeError, ValueError):
                continue
            row = out.setdefault(wid, {})
            cos = c.get("cosine_similarity")
            if cos is None:
                cos = c.get("cosine")
            if cos is not None:
                try:
                    v = float(cos)
                    prev = row.get("cosine")
                    if prev is None or v > prev:
                        row["cosine"] = v
                except (TypeError, ValueError):
                    pass
            if c.get("xgb_score") is not None:
                try:
                    v = float(c["xgb_score"])
                    prev = row.get("xgb_score")
                    if prev is None or v > prev:
                        row["xgb_score"] = v
                except (TypeError, ValueError):
                    pass
    return out


def _report_target_wine_id(
    *,
    matched_id: int | None,
    manual_id: int | None,
    hist_fp: int,
) -> int | None:
    """Целевое вино для cos+/xgb+: manual, иначе matched без FP."""
    if manual_id is not None:
        return int(manual_id)
    if matched_id is not None and int(hist_fp or 0) == 0:
        return int(matched_id)
    return None


def _timing_histogram(seconds_list: list[float]) -> ScanHistoryReportTimingDist:
    """Гистограмма floor(sec): только секунды, которые встречались в выборке."""
    from collections import Counter
    from statistics import mean, median

    if not seconds_list:
        return ScanHistoryReportTimingDist(n=0, bins=[])

    int_secs = [max(0, int(s)) for s in seconds_list]
    counts: Counter[int] = Counter(int_secs)
    n = len(int_secs)
    bins: list[ScanHistoryReportTimingBin] = []
    for sec in sorted(counts):
        cnt = int(counts[sec])
        bins.append(
            ScanHistoryReportTimingBin(
                seconds=sec,
                n=cnt,
                pct=round(100.0 * cnt / n, 2) if n else 0.0,
            )
        )
    return ScanHistoryReportTimingDist(
        n=n,
        mean_sec=round(float(mean(seconds_list)), 3),
        median_sec=round(float(median(seconds_list)), 3),
        min_sec=round(float(min(seconds_list)), 3),
        max_sec=round(float(max(seconds_list)), 3),
        bins=bins,
    )


def _row_total_ms(row: SearchPhoto, status: dict) -> float | None:
    ms = row.hist_total_ms
    if ms is None:
        timings = status.get("timings_ms") or {}
        ms = timings.get("total") if isinstance(timings, dict) else None
    try:
        if ms is None:
            return None
        v = float(ms)
        return v if v >= 0 and v == v else None
    except (TypeError, ValueError):
        return None


def _score_histogram(
    values: list[float],
    *,
    metric: str,
    polarity: str,
) -> ScanHistoryReportScoreDist:
    from collections import Counter

    labels = [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)] + ["1.0"]
    counts: Counter[str] = Counter()
    for c in values:
        if c < 0:
            counts["<0.0"] += 1
        elif c >= 1.0:
            counts["1.0"] += 1
        else:
            lo = int(min(c, 0.999999) * 10) / 10
            counts[f"{lo:.1f}–{lo + 0.1:.1f}"] += 1
    n = len(values)
    bins = []
    for lab in labels:
        cnt = int(counts.get(lab, 0))
        bins.append(
            {
                "range": lab,
                "n": cnt,
                "pct": round(100.0 * cnt / n, 2) if n else 0.0,
            }
        )
    if not values:
        return ScanHistoryReportScoreDist(
            metric=metric,
            polarity=polarity,
            n=0,
            bins=bins,
        )
    vals = sorted(values)

    def _pct(p: float) -> float:
        i = min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))
        return float(vals[i])

    return ScanHistoryReportScoreDist(
        metric=metric,
        polarity=polarity,
        n=n,
        bins=bins,
        min=float(vals[0]),
        max=float(vals[-1]),
        mean=float(sum(vals) / len(vals)),
        median=_pct(50),
    )


@router.get("/scan-history/report", response_model=ScanHistoryReportResponse)
def scan_history_report(
    id_from: int = Query(..., ge=1, description="Inclusive start search_photos.id"),
    id_to: int = Query(..., ge=1, description="Inclusive end search_photos.id"),
    db: Session = Depends(get_db),
) -> ScanHistoryReportResponse:
    """Отчёт TP/FP/TN/FN по диапазону id (inclusive)."""
    lo = int(id_from)
    hi = int(id_to)
    if hi < lo:
        lo, hi = hi, lo

    rows = db.scalars(
        select(SearchPhoto)
        .where(SearchPhoto.id >= lo)
        .where(SearchPhoto.id <= hi)
        .order_by(SearchPhoto.id.asc())
    ).all()

    tp = fp = tn = fn = 0
    false_positives: list[ScanHistoryReportErrorItem] = []
    false_negatives: list[ScanHistoryReportErrorItem] = []
    min_id: int | None = None
    max_id: int | None = None
    cos_plus: list[float] = []
    cos_minus: list[float] = []
    xgb_plus: list[float] = []
    xgb_minus: list[float] = []
    duration_sec: list[float] = []

    for row in rows:
        sid = int(row.id)
        min_id = sid if min_id is None else min(min_id, sid)
        max_id = sid if max_id is None else max(max_id, sid)
        status = dict(row.status or {})
        total_ms = _row_total_ms(row, status)
        if total_ms is not None:
            duration_sec.append(total_ms / 1000.0)
        matched = row.hist_wine_id
        if matched is None:
            matched = _history_resolve_matched_wine_id(status)
        manual = row.manual_wines_id
        hfp = int(row.hist_fp or 0)
        hfn = int(row.hist_fn or 0)
        cls, reason = _classify_scan_eval(
            matched_id=matched,
            manual_id=manual,
            hist_fp=hfp,
            hist_fn=hfn,
        )
        if cls == "TP":
            tp += 1
        elif cls == "FP":
            fp += 1
            false_positives.append(
                ScanHistoryReportErrorItem(
                    id=sid,
                    matched_wine_id=int(matched) if matched is not None else None,
                    manual_wines_id=int(manual) if manual is not None else None,
                    hist_fp=hfp,
                    hist_fn=hfn,
                    reason=reason,
                    cls=cls,
                )
            )
        elif cls == "TN":
            tn += 1
        else:
            fn += 1
            false_negatives.append(
                ScanHistoryReportErrorItem(
                    id=sid,
                    matched_wine_id=int(matched) if matched is not None else None,
                    manual_wines_id=int(manual) if manual is not None else None,
                    hist_fp=hfp,
                    hist_fn=hfn,
                    reason=reason,
                    cls=cls,
                )
            )

        pool = _report_candidate_pool(status)
        target = _report_target_wine_id(
            matched_id=matched,
            manual_id=manual,
            hist_fp=hfp,
        )
        if target is not None and target in pool:
            row_sc = pool[target]
            if "cosine" in row_sc:
                cos_plus.append(row_sc["cosine"])
            if "xgb_score" in row_sc:
                xgb_plus.append(row_sc["xgb_score"])
        for wid, row_sc in pool.items():
            if target is not None and wid == target:
                continue
            if "cosine" in row_sc:
                cos_minus.append(row_sc["cosine"])
            if "xgb_score" in row_sc:
                xgb_minus.append(row_sc["xgb_score"])

    n = tp + fp + tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    with_match = sum(
        1
        for row in rows
        if (
            row.hist_wine_id is not None
            or _history_resolve_matched_wine_id(dict(row.status or {})) is not None
        )
    )

    score_dists = [
        _score_histogram(cos_plus, metric="cosine", polarity="plus"),
        _score_histogram(cos_minus, metric="cosine", polarity="minus"),
        _score_histogram(xgb_plus, metric="xgb", polarity="plus"),
        _score_histogram(xgb_minus, metric="xgb", polarity="minus"),
    ]
    timing_dist = _timing_histogram(duration_sec)

    return ScanHistoryReportResponse(
        id_from=lo,
        id_to=hi,
        counts=ScanHistoryReportCounts(
            TP=tp,
            FP=fp,
            TN=tn,
            FN=fn,
            n=n,
            min_id=min_id,
            max_id=max_id,
        ),
        metrics=ScanHistoryReportMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            specificity=_safe_div(tn, tn + fp),
            npv=_safe_div(tn, tn + fn),
            accuracy=_safe_div(tp + tn, n),
            fpr=_safe_div(fp, fp + tn),
            fnr=_safe_div(fn, fn + tp),
            match_rate=_safe_div(with_match, n),
        ),
        false_positives=false_positives,
        false_negatives=false_negatives,
        rules=[
            "matched ≠ manual → FP (независимо от checkbox)",
            "нет matched, есть manual → FN (независимо от checkbox)",
            "иначе hist_fp=1 → FP; hist_fn=1 → FN",
            "иначе matched → TP; иначе TN",
            "cos+/xgb+: manual, иначе matched без FP",
            "cos−/xgb−: остальные кандидаты пула (~20), в т.ч. ошибочный FP-финалист",
        ],
        score_dists=score_dists,
        timing_dist=timing_dist,
    )


@router.get("/scan-history", response_model=ScanHistoryResponse)
def scan_history(
    offset: int = Query(0, ge=0),
    limit: int = Query(40, ge=1, le=100),
    sort: str = Query("id"),
    order: str = Query("desc"),
    flag: list[str] | None = Query(
        None,
        description="Repeatable: positive|negative|fp|fn (OR). Empty = all.",
    ),
    matched_wine_id: int | None = Query(
        None,
        ge=1,
        description="Deprecated: use wine_q",
    ),
    wine_q: str | None = Query(
        None,
        description="Wine id (digits) or slug substring (case-insensitive)",
    ),
    around_id: int | None = Query(
        None,
        ge=1,
        description="Scroll target: page containing this search_photos.id",
    ),
    db: Session = Depends(get_db),
) -> ScanHistoryResponse:
    """Infinite-scroll list of search_photos (SQL filters on hist_* columns)."""
    sort_key = (sort or "id").strip().lower()
    descending = (order or "desc").strip().lower() != "asc"

    flags_raw = flag or []
    flags: set[str] = set()
    for raw in flags_raw:
        for part in str(raw or "").replace(";", ",").split(","):
            key = part.strip().lower().replace(" ", "_").replace("-", "_")
            if key in ("false_positive", "falsepositive"):
                key = "fp"
            elif key in ("false_negative", "falsenegative"):
                key = "fn"
            if key in ("positive", "negative", "fp", "fn"):
                flags.add(key)

    wine_query = (wine_q or "").strip()
    if not wine_query and matched_wine_id is not None:
        wine_query = str(int(matched_wine_id))

    where_parts: list = []
    if flags:
        flag_parts = []
        # Positive = matched wine and not marked as false positive
        if "positive" in flags:
            flag_parts.append(
                and_(SearchPhoto.hist_wine_id.is_not(None), SearchPhoto.hist_fp == 0)
            )
        # Negative = no match and not marked as false negative
        if "negative" in flags:
            flag_parts.append(
                and_(SearchPhoto.hist_wine_id.is_(None), SearchPhoto.hist_fn == 0)
            )
        if "fp" in flags:
            flag_parts.append(SearchPhoto.hist_fp == 1)
        if "fn" in flags:
            flag_parts.append(SearchPhoto.hist_fn == 1)
        if flag_parts:
            where_parts.append(or_(*flag_parts))

    if wine_query:
        if wine_query.isdigit():
            where_parts.append(SearchPhoto.hist_wine_id == int(wine_query))
        else:
            esc = (
                wine_query.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            where_parts.append(SearchPhoto.hist_wine_slug.ilike(f"%{esc}%"))

    sort_col = {
        "id": SearchPhoto.id,
        "created_at": SearchPhoto.created_at,
        "total_ms": SearchPhoto.hist_total_ms,
        "matched_wine_id": SearchPhoto.hist_wine_id,
        "matched_wine_confidence": SearchPhoto.hist_wine_confidence,
        "confidence": SearchPhoto.hist_wine_confidence,
    }.get(sort_key, SearchPhoto.id)

    primary_order = sort_col.desc() if descending else sort_col.asc()
    secondary_order = SearchPhoto.id.desc() if descending else SearchPhoto.id.asc()
    order_by = (
        (primary_order, secondary_order) if sort_key != "id" else (primary_order,)
    )

    base = select(SearchPhoto)
    if where_parts:
        base = base.where(and_(*where_parts))

    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0

    use_offset = offset
    if around_id is not None:
        ranked = (
            select(
                SearchPhoto.id.label("sid"),
                func.row_number().over(order_by=order_by).label("rn"),
            )
            .where(and_(*where_parts) if where_parts else True)
            .subquery()
        )
        rn = db.scalar(select(ranked.c.rn).where(ranked.c.sid == int(around_id)))
        if rn is None:
            return ScanHistoryResponse(
                items=[],
                total=int(total),
                offset=0,
                limit=limit,
                has_more=False,
            )
        use_offset = max(0, int(rn) - 1 - 2)

    stmt = base.order_by(*order_by).offset(use_offset).limit(limit)
    rows = list(db.scalars(stmt).all())

    wine_ids: list[int] = []
    seen_w: set[int] = set()
    raw_tops: dict[int, list[dict]] = {}
    for r in rows:
        st = dict(r.status or {})
        mid = r.hist_wine_id
        if mid is None:
            mid = _history_resolve_matched_wine_id(st)
        if mid is not None:
            mid_i = int(mid)
            if mid_i not in seen_w:
                seen_w.add(mid_i)
                wine_ids.append(mid_i)
            continue
        raw_top = _history_xgb_top_raw(st)
        if raw_top:
            raw_tops[int(r.id)] = raw_top
            for t in raw_top:
                wid = int(t["id"])
                if wid not in seen_w:
                    seen_w.add(wid)
                    wine_ids.append(wid)

    wines_by_id: dict[int, Wine] = {}
    if wine_ids:
        for w in db.scalars(select(Wine).where(Wine.id.in_(wine_ids))).all():
            wines_by_id[int(w.id)] = w
    sitemap_map = _sitemap_image_map(
        db, [w.slug for w in wines_by_id.values() if w.slug]
    )

    items = []
    for r in rows:
        mid = r.hist_wine_id
        if mid is None:
            mid = _history_resolve_matched_wine_id(dict(r.status or {}))
        wine = wines_by_id.get(int(mid)) if mid is not None else None
        xgb_top: list[ScanHistoryXgbTopItem] | None = None
        if mid is None:
            raw_top = raw_tops.get(int(r.id)) or []
            if raw_top:
                xgb_top = _history_build_xgb_top(raw_top, wines_by_id, sitemap_map)
        items.append(_history_item_from_row(r, wine, xgb_top=xgb_top))

    return ScanHistoryResponse(
        items=items,
        total=int(total),
        offset=use_offset,
        limit=limit,
        has_more=use_offset + len(items) < int(total),
    )


@router.get("/findwine/{search_photos_id}", response_model=FindWineResponse)
def get_findwine_result(
    search_photos_id: int,
    db: Session = Depends(get_db),
) -> FindWineResponse:
    """Reload a previous findwine run by search_photos.id (for shareable URL)."""
    row = db.get(SearchPhoto, search_photos_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Поиск не найден")
    status = dict(row.status or {})
    status.setdefault("search_photos_id", row.id)
    # Recover artifact filenames from steps if top-level missing (older rows)
    if not status.get("crops_filename"):
        status["crops_filename"] = (
            (status.get("steps") or {}).get("yolo") or {}
        ).get("crops_filename")
    if not status.get("label_filename"):
        status["label_filename"] = (
            (status.get("steps") or {}).get("label_crop") or {}
        ).get("filename")
    return _findwine_response_from_status(
        status, manual_wines_id=row.manual_wines_id
    )


@router.patch(
    "/findwine/{search_photos_id}/manual-wine",
    response_model=ManualWineOut,
)
def update_findwine_manual_wine(
    search_photos_id: int,
    body: ManualWineUpdate,
    db: Session = Depends(get_db),
) -> ManualWineOut:
    """Сохранить ручную отметку «Соответствие» (одно вино на поиск; null = снять).

    При установке manual: FP если пайплайн нашёл другое вино, FN если ничего
    не нашёл; при совпадении с matched — снять оба флага.
    """
    from sqlalchemy.orm.attributes import flag_modified

    row = db.get(SearchPhoto, search_photos_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Поиск не найден")
    wine_id = body.wine_id
    if wine_id is not None:
        wine_id = int(wine_id)
        if db.get(Wine, wine_id) is None:
            raise HTTPException(status_code=404, detail="Вино не найдено")
    row.manual_wines_id = wine_id

    status = dict(row.status or {})
    matched_id = row.hist_wine_id
    if matched_id is None:
        matched_id = _history_resolve_matched_wine_id(status)
    flags = eval_flags_for_manual_match(
        manual_id=wine_id,
        matched_id=matched_id,
    )
    if flags is not None:
        fp, fn = flags
        status = apply_status_eval_flags(status, fp, fn)
        row.status = status
        flag_modified(row, "status")
        sync_search_photo_history_columns(row, status)

    db.add(row)
    db.commit()
    db.refresh(row)
    st = dict(row.status or {})
    fp = int(row.hist_fp or 0)
    fn = int(row.hist_fn or 0)
    if not fp and not fn:
        # fallback if hist not synced
        from app.scan_history_denorm import history_eval_flags

        fp, fn = history_eval_flags(st)
    return ManualWineOut(
        search_photos_id=row.id,
        manual_wines_id=row.manual_wines_id,
        false_positive=fp,
        false_negative=fn,
    )


@router.get("/settings", response_model=PipelineSettingsOut)
def get_pipeline_settings() -> PipelineSettingsOut:
    return PipelineSettingsOut(**settings_public_view())


@router.put("/settings", response_model=PipelineSettingsOut)
def put_pipeline_settings(body: PipelineSettingsUpdate) -> PipelineSettingsOut:
    patch: dict = {}
    if body.ocr_preprocess is not None:
        patch["ocr_preprocess"] = body.ocr_preprocess
    if body.ocr_engines is not None:
        patch["ocr_engines"] = body.ocr_engines
    if body.text_weights is not None:
        patch["text_weights"] = body.text_weights
    if body.final_weights is not None:
        patch["final_weights"] = body.final_weights
    if body.use_dinov3 is not None:
        patch["use_dinov3"] = body.use_dinov3
    if body.geometry_siglip2 is not None:
        patch["geometry_siglip2"] = body.geometry_siglip2
    if body.geometry_dinov3 is not None:
        patch["geometry_dinov3"] = body.geometry_dinov3
    if body.use_gemini_ocr is not None:
        patch["use_gemini_ocr"] = body.use_gemini_ocr
    if body.use_openai_ocr is not None:
        patch["use_openai_ocr"] = body.use_openai_ocr
    if body.use_deepseek_ocr is not None:
        patch["use_deepseek_ocr"] = body.use_deepseek_ocr
    if body.use_qwen_ocr is not None:
        patch["use_qwen_ocr"] = body.use_qwen_ocr
    if body.use_yandex_ocr is not None:
        patch["use_yandex_ocr"] = body.use_yandex_ocr
    if body.use_google_vision_ocr is not None:
        patch["use_google_vision_ocr"] = body.use_google_vision_ocr
    if body.yolo_fallback_full_image is not None:
        patch["yolo_fallback_full_image"] = body.yolo_fallback_full_image
    if body.use_yolo is not None:
        patch["use_yolo"] = body.use_yolo
    if body.yolo_variant is not None:
        patch["yolo_variant"] = body.yolo_variant
    if body.bottle_min_conf is not None:
        patch["bottle_min_conf"] = body.bottle_min_conf
    if body.use_bottle_orient is not None:
        patch["use_bottle_orient"] = body.use_bottle_orient
    if body.bottle_orient_min_deg is not None:
        patch["bottle_orient_min_deg"] = body.bottle_orient_min_deg
    if body.label_detect_openai is not None:
        patch["label_detect_openai"] = body.label_detect_openai
    if body.label_detect_gemini is not None:
        patch["label_detect_gemini"] = body.label_detect_gemini
    if body.embedding_device is not None:
        device = str(body.embedding_device).strip().lower()
        if device not in {"cpu", "gpu"}:
            raise HTTPException(
                status_code=400,
                detail="embedding_device must be 'cpu' or 'gpu'",
            )
        patch["embedding_device"] = device
    if body.embed_cpu_use_cache is not None:
        patch["embed_cpu_use_cache"] = bool(body.embed_cpu_use_cache)
    if body.normalize_max_side is not None:
        side = int(body.normalize_max_side)
        if side not in {1280, 1024, 800}:
            raise HTTPException(
                status_code=400,
                detail="normalize_max_side must be 1280, 1024, or 800",
            )
        patch["normalize_max_side"] = side
    if body.candidates_top_n is not None:
        top_n = int(body.candidates_top_n)
        if top_n not in {10, 20, 30, 40}:
            raise HTTPException(
                status_code=400,
                detail="candidates_top_n must be 10, 20, 30, or 40",
            )
        patch["candidates_top_n"] = top_n
    if body.hf_start_timeout_sec is not None:
        try:
            tsec = float(body.hf_start_timeout_sec)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="hf_start_timeout_sec must be a number"
            ) from exc
        if tsec < 1 or tsec > 300:
            raise HTTPException(
                status_code=400,
                detail="hf_start_timeout_sec must be between 1 and 300",
            )
        patch["hf_start_timeout_sec"] = tsec
    if body.text_match_methods is not None:
        patch["text_match_methods"] = list(body.text_match_methods)
    if body.final_score_method is not None:
        patch["final_score_method"] = str(body.final_score_method)
    if body.text_match_thresholds is not None:
        patch["text_match_thresholds"] = dict(body.text_match_thresholds)
    if body.empty_ocr_cosine_threshold is not None:
        try:
            cot = float(body.empty_ocr_cosine_threshold)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="empty_ocr_cosine_threshold must be a number",
            ) from exc
        if cot < 0 or cot > 0.99:
            raise HTTPException(
                status_code=400,
                detail="empty_ocr_cosine_threshold must be between 0 and 0.99",
            )
        patch["empty_ocr_cosine_threshold"] = cot
    if body.xgb_dead_max is not None:
        try:
            xdm = float(body.xgb_dead_max)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="xgb_dead_max must be a number",
            ) from exc
        if xdm < 0 or xdm > 0.99:
            raise HTTPException(
                status_code=400,
                detail="xgb_dead_max must be between 0 and 0.99",
            )
        patch["xgb_dead_max"] = xdm
    if body.compute_hsv is not None:
        patch["compute_hsv"] = body.compute_hsv
    if body.use_hsv_filter is not None:
        patch["use_hsv_filter"] = body.use_hsv_filter
    if body.hsv_bhattacharyya_max is not None:
        try:
            hmax = float(body.hsv_bhattacharyya_max)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="hsv_bhattacharyya_max must be a number",
            ) from exc
        if hmax < 0 or hmax > 1:
            raise HTTPException(
                status_code=400,
                detail="hsv_bhattacharyya_max must be between 0 and 1",
            )
        patch["hsv_bhattacharyya_max"] = hmax
    if body.hsv_ignore_high_cosine is not None:
        patch["hsv_ignore_high_cosine"] = body.hsv_ignore_high_cosine
    if body.hsv_ignore_cosine_min is not None:
        try:
            cmin = float(body.hsv_ignore_cosine_min)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="hsv_ignore_cosine_min must be a number",
            ) from exc
        if cmin < 0 or cmin > 0.99:
            raise HTTPException(
                status_code=400,
                detail="hsv_ignore_cosine_min must be between 0 and 0.99",
            )
        patch["hsv_ignore_cosine_min"] = cmin
    if body.use_hsv_hard_reject is not None:
        patch["use_hsv_hard_reject"] = body.use_hsv_hard_reject
    if body.hsv_hard_reject_max is not None:
        try:
            hhard = float(body.hsv_hard_reject_max)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="hsv_hard_reject_max must be a number",
            ) from exc
        if hhard < 0 or hhard > 1:
            raise HTTPException(
                status_code=400,
                detail="hsv_hard_reject_max must be between 0 and 1",
            )
        patch["hsv_hard_reject_max"] = hhard
    if body.hard_reject_ignore_high_scores is not None:
        patch["hard_reject_ignore_high_scores"] = body.hard_reject_ignore_high_scores
    if body.hard_reject_ignore_cosine_min is not None:
        try:
            hr_cos = float(body.hard_reject_ignore_cosine_min)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="hard_reject_ignore_cosine_min must be a number",
            ) from exc
        if hr_cos < 0 or hr_cos > 0.99:
            raise HTTPException(
                status_code=400,
                detail="hard_reject_ignore_cosine_min must be between 0 and 0.99",
            )
        patch["hard_reject_ignore_cosine_min"] = hr_cos
    if body.hard_reject_ignore_xgb_min is not None:
        try:
            hr_xgb = float(body.hard_reject_ignore_xgb_min)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="hard_reject_ignore_xgb_min must be a number",
            ) from exc
        if hr_xgb < 0 or hr_xgb > 1:
            raise HTTPException(
                status_code=400,
                detail="hard_reject_ignore_xgb_min must be between 0 and 1",
            )
        patch["hard_reject_ignore_xgb_min"] = hr_xgb
    if body.compute_color_delta is not None:
        patch["compute_color_delta"] = body.compute_color_delta
    if body.reuse_previous_searches is not None:
        patch["reuse_previous_searches"] = body.reuse_previous_searches
    if body.show_search_details is not None:
        patch["show_search_details"] = body.show_search_details
    if body.final_ocr is not None:
        patch["final_ocr"] = str(body.final_ocr)
    if body.exclusive_use_translit is not None:
        patch["exclusive_use_translit"] = body.exclusive_use_translit
    if body.exclusive_match_spaced is not None:
        patch["exclusive_match_spaced"] = body.exclusive_match_spaced
    if not patch:
        return PipelineSettingsOut(**settings_public_view(get_settings()))
    saved = save_settings(patch)
    return PipelineSettingsOut(**settings_public_view(saved))


@router.get("/search-photos/{filename}")
def get_search_photo(filename: str) -> FileResponse:
    """Serve pipeline artifacts from SEARCH_PHOTO_DIR (original / _crops / _label)."""
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    target = (SEARCH_PHOTO_DIR / safe_name).resolve()
    root = SEARCH_PHOTO_DIR.resolve()
    if not str(target).startswith(str(root)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type or "application/octet-stream")

