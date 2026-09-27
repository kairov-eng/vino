"""Gemini client: Google AI Studio (API key) or Vertex AI (service account).

Env (как label_detect/gemini_client.py):
  GOOGLE_AI_API_VERTEX=1
  GOOGLE_CLOUD_PROJECT=…
  GOOGLE_CLOUD_LOCATION=global   # для Gemini 3.x
  GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

При VERTEX=0: GEMINI_API_KEY / GEMINI_API_KEY_*.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.db import config as app_config

_VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def use_vertex_api() -> bool:
    return bool(getattr(app_config, "GOOGLE_AI_API_VERTEX", False))


def vertex_project() -> str:
    return (getattr(app_config, "GOOGLE_CLOUD_PROJECT", "") or "").strip()


def vertex_location(*, model: str | None = None) -> str:
    """Gemini 3.x на Vertex — через location=global (регион → часто 404)."""
    loc = (getattr(app_config, "GOOGLE_CLOUD_LOCATION", "") or "").strip()
    mid = (
        model
        or getattr(app_config, "GEMINI_MODEL", "")
        or ""
    ).strip().lower()
    needs_global = mid.startswith("gemini-3") or mid.startswith("gemini-3.")
    if needs_global and loc and loc.lower() not in {"global", ""}:
        return "global"
    if loc:
        return loc
    return "global" if needs_global else "us-central1"


def credentials_path() -> Path | None:
    raw = (getattr(app_config, "GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    raw = raw.strip('"').strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (_BACKEND_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS не найден: {path}"
        )
    return path


def apply_application_credentials() -> Path | None:
    path = credentials_path()
    if path is None:
        return None
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    return path


def load_vertex_credentials():
    from google.oauth2 import service_account

    path = apply_application_credentials()
    if path is None:
        raise RuntimeError(
            "GOOGLE_AI_API_VERTEX=1, но не задан GOOGLE_APPLICATION_CREDENTIALS "
            "(путь к service-account.json рядом с backend/.env)."
        )
    return service_account.Credentials.from_service_account_file(
        str(path),
        scopes=list(_VERTEX_SCOPES),
    )


def make_genai_client(
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> Any:
    """google.genai.Client: Vertex (без api_key) или AI Studio."""
    from google import genai

    if use_vertex_api():
        return _make_vertex_client(model=model)
    return _make_ai_studio_client(api_key=api_key)


def _make_ai_studio_client(*, api_key: str | None) -> Any:
    from google import genai

    key = (api_key or "").strip()
    if not key:
        raise RuntimeError(
            "Нет API key для Google AI Studio. "
            "Задайте GEMINI_API_KEY / GEMINI_API_KEY_* или GOOGLE_AI_API_VERTEX=1."
        )
    return genai.Client(api_key=key)


def _make_vertex_client(*, model: str | None = None) -> Any:
    from google import genai

    project = vertex_project()
    location = vertex_location(model=model)
    if not project:
        raise RuntimeError(
            "GOOGLE_AI_API_VERTEX=1, но не задан GOOGLE_CLOUD_PROJECT"
        )
    creds = load_vertex_credentials()
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=creds,
    )


def describe_backend(*, api_key: str | None = None, model: str | None = None) -> str:
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
