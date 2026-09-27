FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    TORCH_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    CRENC_MODEL_DIR=/model

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.8.0 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8094

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=8 \
  CMD curl -fsS http://127.0.0.1:8094/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8094", "--workers", "1"]
