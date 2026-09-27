# Production image for vino_backend (FastAPI findwine).
# Build from repo root: docker compose -f distrib/docker-compose.yml build
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first (avoids multi‑GB CUDA wheels; disk on aidispatcher is tight).
# Prod requirements omit easyocr/surya/transformers (optional at runtime).
COPY backend/requirements.prod.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app
COPY backend/alembic /app/alembic
COPY backend/alembic.ini /app/alembic.ini
COPY backend/pipeline_settings.json /app/pipeline_settings.json

# Weights & media are mounted at runtime (see docker-compose volumes).
RUN mkdir -p /app/models /data/media /data/search_photos /cache/huggingface

EXPOSE 8092

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8092/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1"]
