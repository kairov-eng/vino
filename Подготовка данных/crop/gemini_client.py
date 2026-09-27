# =============================================================================
# gemini_client.py — общий клиент Gemini: AI Studio (API key) или Vertex AI
# =============================================================================
#
# Переключение в .env:
#   GOOGLE_AI_API_VERTEX=1
#   GOOGLE_CLOUD_PROJECT=your-gcp-project
#   GOOGLE_CLOUD_LOCATION=global          # для Gemini 3.x нужен global, не us-central1
#   GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
#
# При VERTEX=0 (по умолчанию):
#   GEMINI_API_KEY_1 / GEMINI_API_KEY — Google AI Studio.
#
# Vertex инициализация (как в доках google-genai):
#   os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "…"
#   client = genai.Client(vertexai=True, project=…, location=…, credentials=…)
#   # без api_key — иначе SDK уходит не в тот endpoint
#
# =============================================================================

"""Клиент Gemini: Developer API (AI Studio) или Vertex AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2 import service_account

SCRIPT_DIR = Path(__file__).resolve().parent
def resolve_env_path() -> Path:
    """Find .env: local folder → Подготовка данных → backend/.env."""
    here = Path(__file__).resolve().parent
    for c in (
        here / ".env",
        here.parent / ".env",
        here.parents[1] / "backend" / ".env",
    ):
        if c.is_file():
            return c
    return here / ".env"

ENV_PATH = resolve_env_path()

_VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def _load_env() -> None:
    load_dotenv(ENV_PATH, override=True)


def _env_truthy(name: str, default: str = "0") -> bool:
    val = (os.getenv(name) or default).strip().lower()
    return val in {"1", "true", "yes", "on"}


def use_vertex_api() -> bool:
    """True, если GOOGLE_AI_API_VERTEX=1 — ходить в Vertex AI, не в AI Studio."""
    _load_env()
    return _env_truthy("GOOGLE_AI_API_VERTEX", "0")


def vertex_project() -> str:
    _load_env()
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or ""
    ).strip()


def vertex_location(*, model: str | None = None) -> str:
    """
    Регион Vertex.
    Gemini 3.x на Vertex доступны через location=global (us-central1 → 404).
    """
    _load_env()
    loc = (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("VERTEX_LOCATION")
        or ""
    ).strip()
    mid = (model or os.getenv("GEMINI_MODEL") or "").strip().lower()
    needs_global = mid.startswith("gemini-3") or mid.startswith("gemini-3.")
    if needs_global and loc and loc.lower() not in {"global", ""}:
        # региональный endpoint для 3.x часто даёт 404 NOT_FOUND
        return "global"
    if loc:
        return loc
    return "global" if needs_global else "us-central1"


def credentials_path() -> Path | None:
    _load_env()
    raw = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip().strip('"')
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS не найден: {path} "
            f"(проверьте путь в {ENV_PATH})"
        )
    return path.resolve()


def apply_application_credentials() -> Path | None:
    """Прописывает SA JSON в os.environ (как ожидает google.auth / genai)."""
    path = credentials_path()
    if path is None:
        return None
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    return path


def load_vertex_credentials():
    """Service Account credentials из GOOGLE_APPLICATION_CREDENTIALS."""
    path = apply_application_credentials()
    if path is None:
        raise RuntimeError(
            "GOOGLE_AI_API_VERTEX=1, но не задан GOOGLE_APPLICATION_CREDENTIALS "
            f"(путь к service-account.json в {ENV_PATH}).\n"
            "Либо: gcloud auth application-default login"
        )
    return service_account.Credentials.from_service_account_file(
        str(path),
        scopes=list(_VERTEX_SCOPES),
    )


def make_genai_client(
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> genai.Client:
    """
    Создаёт клиент google-genai.

    - GOOGLE_AI_API_VERTEX=1 → Vertex AI (vertexai=True + project + location + SA).
      api_key НЕ передаётся (иначе SDK может уйти не на Vertex endpoint).
    - иначе → Gemini Developer API (api_key).
    """
    _load_env()
    if use_vertex_api():
        return _make_vertex_client(model=model)
    return _make_ai_studio_client(api_key=api_key)


def _make_ai_studio_client(*, api_key: str | None) -> genai.Client:
    key = (api_key or "").strip()
    if not key:
        raise RuntimeError(
            "Нет API key для Google AI Studio. "
            "Задайте GEMINI_API_KEY_1 или включите GOOGLE_AI_API_VERTEX=1."
        )
    return genai.Client(api_key=key)


def _make_vertex_client(*, model: str | None = None) -> genai.Client:
    project = vertex_project()
    location = vertex_location(model=model)
    if not project:
        raise RuntimeError(
            "GOOGLE_AI_API_VERTEX=1, но не задан GOOGLE_CLOUD_PROJECT "
            f"(см. {ENV_PATH})"
        )
    creds = load_vertex_credentials()
    # Важно: только vertexai=True + project + location + credentials, БЕЗ api_key
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=creds,
    )


def describe_backend(*, api_key: str | None = None, model: str | None = None) -> str:
    """Короткая строка для логов."""
    if use_vertex_api():
        cred = credentials_path()
        cred_s = cred.name if cred else "(ADC/default)"
        return (
            f"Vertex AI  project={vertex_project() or '?'}  "
            f"location={vertex_location(model=model)}  "
            f"credentials={cred_s}  vertexai=True"
        )
    tail = f"...{api_key[-6:]}" if api_key and len(api_key) >= 6 else "(no key)"
    return f"AI Studio (API key {tail})"


def json_generate_configs() -> list[types.GenerateContentConfig]:
    """Конфиги с fallback: thinking_budget → без thinking."""
    return [
        types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    ]


def generate_json(
    client: genai.Client,
    *,
    model: str,
    contents: Any,
) -> Any:
    """
    generate_content с response JSON.
    contents — str или list частей (текст / Part.from_bytes).
    """
    last_exc: Exception | None = None
    for cfg in json_generate_configs():
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=cfg,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    assert last_exc is not None
    raise last_exc
