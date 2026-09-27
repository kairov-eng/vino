#!/usr/bin/env bash
# One-shot deploy helper on aidispatcher (vino-svoe.online).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"

sudo mkdir -p /var/lib/vino-svoe/{media,search_photos,models,hf-cache}
sudo chown -R "$USER":"$USER" /var/lib/vino-svoe || true

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo "Created $ROOT/.env — edit secrets before continuing."
  exit 1
fi

cd "$REPO"
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d --build
docker compose -f distrib/docker-compose.yml ps
curl -fsS http://127.0.0.1:8092/api/health || true
curl -fsS -o /dev/null -w "frontend:%{http_code}\n" http://127.0.0.1:8091/ || true
