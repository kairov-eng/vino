from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import mimetypes
from pathlib import Path

from app.db.config import CORS_ORIGINS, MEDIA_ROOT, MEDIA_URL_PATH
from app.database import SessionLocal
from app.media import media_stats
from app.pipeline.analogs import load_analog_catalog
from app.pipeline.exclusive_lexicon import (
    load_grape_variety_lexicon,
    load_static_lexicons,
    load_winery_lexicon,
)
from app.pipeline.crenc_match import start_crenc_warmup_background
from app.pipeline.ocr import start_ocr_warmup_background
from app.pipeline.ocr_match import warm_producers_regions_cache
from app.pipeline.ocr_match_v2 import build_catalog_idf
from app.pipeline.xgb_match import start_xgb_warmup_background
from app.pipeline.runtime_settings import get_settings
from app.pipeline.yolo_detect import start_yolo_warmup_background
from app.routers import eval_router, router
from app.schemas import SiteAuthRequest, SiteAuthResponse
from app.site_auth import admin_password_required, is_admin_password

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("image/png", ".png")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # UI-настройки из pipeline_settings.json → app_config (не оставлять .env-дефолты)
    settings0 = get_settings()
    # Справочники exclusive_lexicon (category/type JSON + сорта/винодельни из БД)
    try:
        load_static_lexicons()
        with SessionLocal() as db:
            load_grape_variety_lexicon(db)
            load_winery_lexicon(db)
            # После лексиконов: каталог для аналогов использует их матчеры
            load_analog_catalog(db)
            # Soft IDF + producers/regions — чтобы первый поиск не платил cold
            warm_producers_regions_cache(db)
            build_catalog_idf(
                db,
                exclude_name=bool(settings0.get("fast_text_match", True)),
            )
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger("vino.exclusive_lexicon").exception(
            "failed to load exclusive lexicons at startup"
        )
    start_ocr_warmup_background()
    start_yolo_warmup_background()
    start_xgb_warmup_background()
    start_crenc_warmup_background()
    yield


app = FastAPI(title="Vino Svoe API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(eval_router)

MEDIA_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "media": media_stats()}


@app.get("/api/admin-auth", response_model=SiteAuthResponse)
def admin_auth_status() -> SiteAuthResponse:
    """Whether an admin password is required to change pipeline settings."""
    required = admin_password_required()
    return SiteAuthResponse(
        ok=True,
        role="admin" if not required else None,
        enabled=required,
    )


@app.post("/api/admin-auth", response_model=SiteAuthResponse)
def admin_auth_verify(body: SiteAuthRequest) -> SiteAuthResponse:
    """Validate admin password for settings edit unlock. Client stores it in a cookie."""
    if not admin_password_required():
        return SiteAuthResponse(ok=True, role="admin", enabled=False)
    if not is_admin_password(body.password):
        raise HTTPException(status_code=401, detail="invalid password")
    return SiteAuthResponse(ok=True, role="admin", enabled=True)


@app.get(f"{MEDIA_URL_PATH}/{{file_path:path}}")
def serve_media(file_path: str):
    # Prevent path traversal.
    # Prod prefers nginx static (/media via vino_frontend); this is local/dev fallback.
    target = (MEDIA_ROOT / file_path).resolve()
    if not str(target).startswith(str(MEDIA_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type, _ = mimetypes.guess_type(str(target))
    if not media_type:
        suffix = target.suffix.lower()
        media_type = {
            ".webp": "image/webp",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }.get(suffix, "application/octet-stream")
    return FileResponse(
        target,
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=2592000, immutable",
        },
    )
