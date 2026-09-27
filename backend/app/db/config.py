from pathlib import Path

from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[2]
# override=True: после reload по изменению .env новые значения побеждают старые из окружения
load_dotenv(ROOT / ".env", override=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgresql:password@127.0.0.1:5432/vino",
)
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://127.0.0.1:8091,http://localhost:8091"
    ).split(",")
    if o.strip()
]

# Site access gate (cookie vino_site_password). Empty = gate disabled.
PASSWORD_ADMIN = (os.getenv("password_admin") or "").strip()
PASSWORD_USER = (os.getenv("password_user") or "").strip()
SITE_ACCESS_ENABLED = bool(PASSWORD_ADMIN or PASSWORD_USER)

# Local wine images. Default: C:\dev\Vino2026\uploads (next to vino-svoe/)
_default_media = ROOT.parent / "uploads"
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", str(_default_media))).resolve()
MEDIA_URL_PATH = os.getenv("MEDIA_URL_PATH", "/media")

# Search pipeline: uploaded query photos
_default_photo_logs = ROOT / "logs" / "photo"
SEARCH_PHOTO_DIR = Path(
    os.getenv("SEARCH_PHOTO_DIR", str(_default_photo_logs))
).resolve()

# YOLO label detector weights (default: backend/models/yolo/label_detect.pt)
_default_yolo = ROOT / "models" / "yolo" / "label_detect.pt"
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(_default_yolo)).strip()
_default_yolo_bottle_label = ROOT / "models" / "yolo_bottle_label" / "best.pt"
YOLO_BOTTLE_LABEL_MODEL_PATH = os.getenv(
    "YOLO_BOTTLE_LABEL_MODEL_PATH", str(_default_yolo_bottle_label)
).strip()

# XGBoost OCR↔label matcher (dir with model.json + feature_names.json)
_default_xgb = ROOT / "models" / "xgboost_text_matcher_synthetic_40_60"


def _resolve_model_dir(env_key: str, default: Path) -> Path:
    raw = (os.getenv(env_key) or "").strip()
    if not raw:
        return default.resolve()
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


XGB_MODEL_DIR = _resolve_model_dir("XGB_MODEL_DIR", _default_xgb)
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
# label | bottle_label — runtime via pipeline_settings.json (yolo_variant)
YOLO_VARIANT = (os.getenv("YOLO_VARIANT", "label") or "label").strip().lower()
if YOLO_VARIANT not in {"label", "bottle_label"}:
    YOLO_VARIANT = "label"
# "0" / "cuda:0" / "cpu" — при отсутствии CUDA автоматически cpu
YOLO_DEVICE = (os.getenv("YOLO_DEVICE", "0") or "0").strip()
YOLO_HALF = (os.getenv("YOLO_HALF", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Remote embedding services (multipart POST …/v1/embed)
SIGLIP2_ENDPOINT = os.getenv(
    "SIGLIP2_ENDPOINT", "https://vino-svoe.online/api_siglip2"
).rstrip("/")
DINOV3_ENDPOINT = os.getenv(
    "DINOV3_ENDPOINT", "https://vino-svoe.online/api_dinov3"
).rstrip("/")
# Optional extra HTTP for DINOv3 only. SigLIP2 local = in-process (SIGLIP2_LOCAL_*).
SIGLIP2_ENDPOINT_LOCAL = os.getenv("SIGLIP2_ENDPOINT_LOCAL", "").rstrip("/")
DINOV3_ENDPOINT_LOCAL = os.getenv("DINOV3_ENDPOINT_LOCAL", "").rstrip("/")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_TIMEOUT_SEC = float(os.getenv("EMBED_TIMEOUT_SEC", "120"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
# In-process SigLIP2 fallback after remote GPU/CPU (transformers, not Docker)
_siglip_local_raw = (os.getenv("SIGLIP2_LOCAL_ENABLED", "1") or "1").strip().lower()
SIGLIP2_LOCAL_ENABLED = _siglip_local_raw in {"1", "true", "yes", "on", "y"}
# Hub id used when downloading into SIGLIP2_LOCAL_DIR
SIGLIP2_LOCAL_MODEL = (
    os.getenv("SIGLIP2_LOCAL_MODEL", "google/siglip2-base-patch16-384") or ""
).strip() or "google/siglip2-base-patch16-384"
_default_siglip2_dir = ROOT / "models" / "Siglip2"
SIGLIP2_LOCAL_DIR = _resolve_model_dir("SIGLIP2_LOCAL_DIR", _default_siglip2_dir)
SIGLIP2_LOCAL_DEVICE = (
    os.getenv("SIGLIP2_LOCAL_DEVICE", "cpu") or "cpu"
).strip().lower() or "cpu"
SIGLIP2_LOCAL_TORCH_THREADS = int(
    os.getenv("SIGLIP2_LOCAL_TORCH_THREADS", "2") or "2"
)
# HF SigLIP2 payload: JPEG short side (model is patch16-384)
HF_EMBED_SHORT_SIDE = int(os.getenv("HF_EMBED_SHORT_SIDE", "384") or "384")
if HF_EMBED_SHORT_SIDE < 64:
    HF_EMBED_SHORT_SIDE = 384
# CrossEncoder matcher (CPU server) — same X-API-Key as siglip2
CRENC_ENDPOINT = os.getenv(
    "CRENC_ENDPOINT", "https://vino-svoe.online/api_cross_encoder_matcher"
).rstrip("/")
CRENC_TIMEOUT_SEC = float(
    os.getenv("CRENC_TIMEOUT_SEC") or os.getenv("EMBED_TIMEOUT_SEC") or "120"
)
# cpu = own server (SIGLIP2_ENDPOINT / DINOV3_ENDPOINT); gpu = Hugging Face endpoints
EMBEDDING_DEVICE = (os.getenv("EMBEDDING_DEVICE", "cpu") or "cpu").strip().lower()
if EMBEDDING_DEVICE not in {"cpu", "gpu"}:
    EMBEDDING_DEVICE = "cpu"

# Hugging Face Inference Endpoints (used when EMBEDDING_DEVICE=gpu)
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "").strip().rstrip("/")
HF_SIGLIP2_ENDPOINT = (
    os.getenv("HF_SIGLIP2_ENDPOINT") or HF_ENDPOINT or ""
).strip().rstrip("/")
HF_DINOV3_ENDPOINT = os.getenv("HF_DINOV3_ENDPOINT", "").strip().rstrip("/")
HF_ACCESS_TOKEN = (
    os.getenv("HF_ACCESS_TOKEN") or os.getenv("HF_TOKEN") or ""
).strip()
HF_TIMEOUT_SEC = float(
    os.getenv("HF_TIMEOUT_SEC") or os.getenv("EMBED_TIMEOUT_SEC") or "120"
)
HF_NAMESPACE = os.getenv("HF_NAMESPACE", "").strip()
HF_SIGLIP2_ENDPOINT_NAME = os.getenv("HF_SIGLIP2_ENDPOINT_NAME", "").strip()
HF_DINOV3_ENDPOINT_NAME = os.getenv("HF_DINOV3_ENDPOINT_NAME", "").strip()
# Qwen2.5-VL OCR (Inference Endpoint, OpenAI-compatible /v1/chat/completions)
HF_QWEN_OCR_ENDPOINT = os.getenv("HF_QWEN_OCR_ENDPOINT", "").strip().rstrip("/")
HF_QWEN_OCR_ENDPOINT_NAME = os.getenv("HF_QWEN_OCR_ENDPOINT_NAME", "").strip()
HF_QWEN_OCR_MODEL = (
    os.getenv("HF_QWEN_OCR_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct") or ""
).strip() or "Qwen/Qwen2.5-VL-3B-Instruct"
HF_QWEN_OCR_TIMEOUT_SEC = float(
    os.getenv("HF_QWEN_OCR_TIMEOUT_SEC")
    or os.getenv("HF_TIMEOUT_SEC")
    or "120"
)
# Сколько ждать resume HF перед fallback на свой CPU endpoint
HF_START_TIMEOUT_SEC = float(os.getenv("HF_START_TIMEOUT_SEC", "15"))

# Work JPEG long-side cap on normalize (findwine)
_NORMALIZE_ALLOWED = (1280, 1024, 800)
try:
    _nms = int(os.getenv("NORMALIZE_MAX_SIDE", "1024"))
except ValueError:
    _nms = 1024
NORMALIZE_MAX_SIDE = _nms if _nms in _NORMALIZE_ALLOWED else 1024

# Budget for normalized work JPEG (original upload stays as-is)
try:
    WORK_JPEG_MAX_BYTES = int(os.getenv("WORK_JPEG_MAX_BYTES", str(1_000_000)))
except ValueError:
    WORK_JPEG_MAX_BYTES = 1_000_000
WORK_JPEG_MAX_BYTES = max(100_000, min(5_000_000, WORK_JPEG_MAX_BYTES))

# PP-OCRv5 via rapidocr-onnxruntime (optional custom ONNX paths for eslav)
OCR_DET_MODEL_PATH = os.getenv("OCR_DET_MODEL_PATH", "").strip()
OCR_REC_MODEL_PATH = os.getenv("OCR_REC_MODEL_PATH", "").strip()
OCR_TIMEOUT_MS = int(os.getenv("OCR_TIMEOUT_MS", "700"))
# Comma-separated OCR engines: rapid, easy, surya, tess (order = run order)
OCR_ENGINES = [
    e.strip().lower()
    for e in os.getenv("OCR_ENGINES", "rapid,easy,surya,tess").split(",")
    if e.strip()
]
# Comma-separated preprocess variants: A,B,C,D
OCR_PREPROCESS = [
    e.strip().upper()
    for e in os.getenv("OCR_PREPROCESS", "A,B,C,D").split(",")
    if e.strip().upper() in {"A", "B", "C", "D"}
] or ["A", "B", "C", "D"]
# Hard wall-clock budget for the whole OCR step (keeps API responsive)
OCR_BUDGET_SEC = float(os.getenv("OCR_BUDGET_SEC", "180"))
# Per-engine wall clock after the worker starts (not queue wait)
OCR_ENGINE_TIMEOUT_SEC = {
    "rapid": float(os.getenv("OCR_TIMEOUT_RAPID_SEC", "60")),
    "easy": float(os.getenv("OCR_TIMEOUT_EASY_SEC", "60")),
    "tess": float(os.getenv("OCR_TIMEOUT_TESS_SEC", "60")),
    # CPU cold-start: load + detect + recognize часто 60–100 с
    "surya": float(os.getenv("OCR_TIMEOUT_SURYA_SEC", "120")),
}
# Downscale before heavy Torch OCR (CPU); smaller → быстрее
OCR_MAX_SIDE_EASY = int(os.getenv("OCR_MAX_SIDE_EASY", "512"))
OCR_MAX_SIDE_SURYA = int(os.getenv("OCR_MAX_SIDE_SURYA", "448"))


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Easy/Surya только на preprocess A (не на upscale B–D)
OCR_HEAVY_ONLY_PREPROCESS_A = _env_flag("OCR_HEAVY_ONLY_PREPROCESS_A", True)

# TextScore weights (must sum ~1.0; renormalized at runtime)
TEXT_WEIGHT_WINE = float(os.getenv("TEXT_WEIGHT_WINE", "0.35"))
TEXT_WEIGHT_PRODUCER = float(os.getenv("TEXT_WEIGHT_PRODUCER", "0.28"))
TEXT_WEIGHT_COLOR = float(os.getenv("TEXT_WEIGHT_COLOR", "0.14"))
TEXT_WEIGHT_TYPE = float(os.getenv("TEXT_WEIGHT_TYPE", "0.08"))
TEXT_WEIGHT_GRAPE = float(os.getenv("TEXT_WEIGHT_GRAPE", "0.12"))
TEXT_WEIGHT_VINTAGE = float(os.getenv("TEXT_WEIGHT_VINTAGE", "0.12"))
TEXT_WEIGHT_REGION = float(os.getenv("TEXT_WEIGHT_REGION", "0.08"))
TEXT_WEIGHT_OTHER = float(os.getenv("TEXT_WEIGHT_OTHER", "0.05"))
TEXT_WEIGHT_BRAND = float(os.getenv("TEXT_WEIGHT_BRAND", "0.12"))

# FinalScore = w_text×TextScore(OCR) + w_cos×Cosine(embedding)
FINAL_SCORE_W_TEXT = float(os.getenv("FINAL_SCORE_W_TEXT", "0.45"))
FINAL_SCORE_W_COS = float(os.getenv("FINAL_SCORE_W_COS", "0.55"))


# Embedding / geometry pipeline toggles (also in pipeline_settings.json)
USE_DINOV3 = _env_flag("USE_DINOV3", True)
GEOMETRY_SIGLIP2 = _env_flag("GEOMETRY_SIGLIP2", True)
GEOMETRY_DINOV3 = _env_flag("GEOMETRY_DINOV3", True)
# Redis cache on CPU embed endpoints (use_cache=1 for siglip2 / dinov3)
EMBED_CPU_USE_CACHE = _env_flag("EMBED_CPU_USE_CACHE", False)
# Если YOLO не нашёл этикетку — брать весь кадр как _label
YOLO_FALLBACK_FULL_IMAGE = _env_flag("YOLO_FALLBACK_FULL_IMAGE", True)
# YOLO детектор этикетки; false → LLM detect+OCR (OpenAI/Gemini) вместо YOLO
USE_YOLO = _env_flag("USE_YOLO", True)
# При выкл. YOLO: кто ищет рамку этикетки (+OCR в том же запросе)
LABEL_DETECT_OPENAI = _env_flag("LABEL_DETECT_OPENAI", True)
LABEL_DETECT_GEMINI = _env_flag("LABEL_DETECT_GEMINI", True)

# Google Gemini vision OCR (single original label image → JSON)
USE_GEMINI_OCR = _env_flag("USE_GEMINI_OCR", True)
# Прямой вызов Google (AI Studio API key или Vertex SA), иначе HTTP-прокси.
# Vertex: GOOGLE_AI_API_VERTEX=1 + GOOGLE_CLOUD_PROJECT + credentials JSON.
GEMINI_PROXY_URL = os.getenv(
    "GEMINI_PROXY_URL", "https://vino-svoe.online"
).strip().rstrip("/")
GEMINI_FORCE_PROXY = _env_flag("GEMINI_FORCE_PROXY", False)
GOOGLE_AI_API_VERTEX = _env_flag("GOOGLE_AI_API_VERTEX", False)
GOOGLE_CLOUD_PROJECT = (
    os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("GCLOUD_PROJECT")
    or os.getenv("GCP_PROJECT")
    or ""
).strip()
GOOGLE_CLOUD_LOCATION = (
    os.getenv("GOOGLE_CLOUD_LOCATION")
    or os.getenv("VERTEX_LOCATION")
    or "global"
).strip() or "global"
GOOGLE_APPLICATION_CREDENTIALS = (
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or ""
).strip().strip('"')


def _load_gemini_api_keys() -> list[str]:
    keys: list[str] = []
    i = 1
    while True:
        val = (os.getenv(f"GEMINI_API_KEY_{i}") or "").strip()
        if not val:
            break
        keys.append(val)
        i += 1
    legacy = (os.getenv("GEMINI_API_KEY") or "").strip()
    if legacy and legacy not in keys:
        keys.append(legacy)
    return keys


GEMINI_API_KEYS = _load_gemini_api_keys()
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
GEMINI_OCR_MAX_SIDE = int(os.getenv("GEMINI_OCR_MAX_SIDE", "1280"))
GEMINI_OCR_MAX_BYTES = int(os.getenv("GEMINI_OCR_MAX_BYTES", "3500000"))
GEMINI_OCR_TIMEOUT_SEC = float(os.getenv("GEMINI_OCR_TIMEOUT_SEC", "120"))

# OpenAI vision OCR (общий промпт с Gemini — см. OCR_PROMPT_PATH)
USE_OPENAI_OCR = _env_flag("USE_OPENAI_OCR", True)
OPENAI_KEY = (
    os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or ""
).strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_OCR_TIMEOUT_SEC = float(os.getenv("OPENAI_OCR_TIMEOUT_SEC", "120"))

# DeepSeek vision OCR (OpenAI-compatible; общий промпт — OCR_PROMPT_PATH)
USE_DEEPSEEK_OCR = _env_flag("USE_DEEPSEEK_OCR", True)
DEEPSEEK_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY") or ""
).strip()
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
).strip().rstrip("/")
DEEPSEEK_MODEL = (
    os.getenv("DEEPSEEK_MODEL", "deepseek-flash") or ""
).strip() or "deepseek-flash"
DEEPSEEK_OCR_TIMEOUT_SEC = float(os.getenv("DEEPSEEK_OCR_TIMEOUT_SEC", "120"))

# Yandex Vision OCR (recognizeText; lines only)
USE_YANDEX_OCR = _env_flag("USE_YANDEX_OCR", True)
YANDEX_API_KEY = (os.getenv("YANDEX_API_KEY") or "").strip()
YANDEX_FOLDER_ID = (os.getenv("YANDEX_FOLDER_ID") or "").strip()
YANDEX_OCR_URL = (
    os.getenv(
        "YANDEX_OCR_URL",
        "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText",
    )
    or ""
).strip()
YANDEX_OCR_MODEL = (os.getenv("YANDEX_OCR_MODEL", "page") or "").strip() or "page"
YANDEX_OCR_LANGUAGES = (
    os.getenv("YANDEX_OCR_LANGUAGES", "*") or ""
).strip() or "*"
YANDEX_OCR_TIMEOUT_SEC = float(os.getenv("YANDEX_OCR_TIMEOUT_SEC", "120"))

# Google Cloud Vision OCR (тот же SA: GOOGLE_APPLICATION_CREDENTIALS)
USE_GOOGLE_VISION_OCR = _env_flag("USE_GOOGLE_VISION_OCR", True)
GOOGLE_VISION_FEATURE = (
    os.getenv("GOOGLE_VISION_FEATURE", "DOCUMENT_TEXT_DETECTION") or ""
).strip() or "DOCUMENT_TEXT_DETECTION"
GOOGLE_VISION_LANGUAGES = (os.getenv("GOOGLE_VISION_LANGUAGES") or "").strip()
GOOGLE_VISION_TIMEOUT_SEC = float(os.getenv("GOOGLE_VISION_TIMEOUT_SEC", "120"))

# Hugging Face Qwen2.5-VL OCR (endpoint URL / model — в блоке HF_* выше)
USE_QWEN_OCR = _env_flag("USE_QWEN_OCR", False)

# Промпты LLM (общие для Gemini и OpenAI). Путь абсолютный или от корня backend/
_PIPELINE_PROMPTS = ROOT / "app" / "pipeline"


def _prompt_path(env_key: str, default_name: str) -> Path:
    raw = (os.getenv(env_key) or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (ROOT / p).resolve()
    return (_PIPELINE_PROMPTS / default_name).resolve()


LABEL_OCR_PROMPT_PATH = _prompt_path("LABEL_OCR_PROMPT_PATH", "label_ocr_prompt.txt")
OCR_PROMPT_PATH = _prompt_path("OCR_PROMPT_PATH", "ocr_prompt.txt")
