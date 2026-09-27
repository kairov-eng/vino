#!/usr/bin/env bash
# Render nginx config from template (HTTP or HTTPS if cert exists).
# Usage: ./apply-nginx-config.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
set -a
# shellcheck source=/dev/null
source ./.env
set +a

DOMAIN="${DOMAIN:-aidispatcher.online}"
LANGFUSE_DOMAIN="${LANGFUSE_DOMAIN:-langfuse.${DOMAIN}}"
VINO_SVOE_DOMAIN="${VINO_SVOE_DOMAIN:-vino-svoe.online}"
CERT_DIR="../certbot/conf/live/${DOMAIN}"
LANGFUSE_CERT_DIR="../certbot/conf/live/${LANGFUSE_DOMAIN}"
VINO_SVOE_CERT_DIR="../certbot/conf/live/${VINO_SVOE_DOMAIN}"
OUT="./nginx/conf.d/default.conf"
LANGFUSE_OUT="./nginx/conf.d/langfuse.conf"
VINO_SVOE_OUT="./nginx/conf.d/vino-svoe.conf"

mkdir -p ./nginx/conf.d ../certbot/www ../certbot/conf

if [[ -f "${CERT_DIR}/fullchain.pem" && -f "${CERT_DIR}/privkey.pem" ]]; then
  TEMPLATE="./nginx/https.conf.template"
  echo "Using HTTPS nginx config (cert found for ${DOMAIN})"
else
  TEMPLATE="./nginx/http.conf.template"
  echo "Using HTTP nginx config (no cert yet for ${DOMAIN})"
fi

# Replace ${DOMAIN} placeholders
sed "s/\${DOMAIN}/${DOMAIN}/g" "$TEMPLATE" > "$OUT"
echo "Wrote $OUT"

if [[ -f "${LANGFUSE_CERT_DIR}/fullchain.pem" && -f "${LANGFUSE_CERT_DIR}/privkey.pem" ]]; then
  LANGFUSE_TEMPLATE="./nginx/langfuse.https.conf.template"
  echo "Using HTTPS nginx config for ${LANGFUSE_DOMAIN}"
else
  LANGFUSE_TEMPLATE="./nginx/langfuse.http.conf.template"
  echo "Using HTTP nginx config for ${LANGFUSE_DOMAIN} (no cert yet)"
fi

sed "s/\${LANGFUSE_DOMAIN}/${LANGFUSE_DOMAIN}/g" "$LANGFUSE_TEMPLATE" > "$LANGFUSE_OUT"
echo "Wrote $LANGFUSE_OUT"

# vino-svoe.online — separate vhost; never serves aidispatcher frontend
if [[ -f ./nginx/vino-svoe.https.conf.template ]]; then
  if [[ -f "${VINO_SVOE_CERT_DIR}/fullchain.pem" && -f "${VINO_SVOE_CERT_DIR}/privkey.pem" ]]; then
    VINO_SVOE_TEMPLATE="./nginx/vino-svoe.https.conf.template"
    echo "Using HTTPS nginx config for ${VINO_SVOE_DOMAIN}"
  else
    VINO_SVOE_TEMPLATE="./nginx/vino-svoe.http.conf.template"
    echo "Using HTTP nginx config for ${VINO_SVOE_DOMAIN} (no cert yet)"
  fi
  sed "s/\${VINO_SVOE_DOMAIN}/${VINO_SVOE_DOMAIN}/g" "$VINO_SVOE_TEMPLATE" > "$VINO_SVOE_OUT"
  echo "Wrote $VINO_SVOE_OUT"
fi
