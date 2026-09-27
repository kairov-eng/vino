#!/usr/bin/env bash
# Issue Let's Encrypt cert for vino-svoe.online via existing certbot webroot.
# Does not modify aidispatcher.online default.conf.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# Expected layout on server: /opt/aidispatcher/distrib/
# This script is copied next to issue-langfuse-cert.sh
cd "$ROOT"

# shellcheck disable=SC1091
set -a
# shellcheck source=/dev/null
source ./.env
set +a

VINO_SVOE_DOMAIN="${VINO_SVOE_DOMAIN:-vino-svoe.online}"
EMAIL="${CERTBOT_EMAIL:-}"
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

if [[ -z "$EMAIL" ]]; then
  echo "Set CERTBOT_EMAIL in distrib/.env" >&2
  exit 1
fi

mkdir -p ../certbot/www ../certbot/conf ./nginx/conf.d

# HTTP-only vhost for ACME (safe before cert exists)
sed "s/\${VINO_SVOE_DOMAIN}/${VINO_SVOE_DOMAIN}/g" \
  ./nginx/vino-svoe.http.conf.template \
  > ./nginx/conf.d/vino-svoe.conf

docker compose --env-file .env up -d nginx
sleep 2
docker compose --env-file .env exec nginx nginx -t
docker compose --env-file .env exec nginx nginx -s reload || \
  docker compose --env-file .env restart nginx

CERTBOT_ARGS=(
  certonly
  --webroot
  --webroot-path=/var/www/certbot
  --email "$EMAIL"
  --agree-tos
  --no-eff-email
  --non-interactive
  -d "$VINO_SVOE_DOMAIN"
  -d "www.$VINO_SVOE_DOMAIN"
)

if [[ "$FORCE" -eq 1 ]]; then
  CERTBOT_ARGS+=(--force-renewal)
fi

echo "Requesting certificate for ${VINO_SVOE_DOMAIN} and www.${VINO_SVOE_DOMAIN} ..."
docker run --rm \
  -v "$(cd .. && pwd)/certbot/www:/var/www/certbot" \
  -v "$(cd .. && pwd)/certbot/conf:/etc/letsencrypt" \
  certbot/certbot \
  "${CERTBOT_ARGS[@]}"

# Switch to HTTPS vhost
sed "s/\${VINO_SVOE_DOMAIN}/${VINO_SVOE_DOMAIN}/g" \
  ./nginx/vino-svoe.https.conf.template \
  > ./nginx/conf.d/vino-svoe.conf

docker compose --env-file .env exec nginx nginx -t
docker compose --env-file .env exec nginx nginx -s reload || \
  docker compose --env-file .env restart nginx

echo
echo "OK: https://${VINO_SVOE_DOMAIN}/health"
echo "Certificates: ../certbot/conf/live/${VINO_SVOE_DOMAIN}/"
