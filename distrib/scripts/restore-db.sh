#!/usr/bin/env bash
# Restore catalog dump into existing Postgres (no search_photos rows).
# Usage:
#   export PGPASSWORD=...
#   ./restore-db.sh postgresql://vino:pass@127.0.0.1:5432/vino
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DSN="${1:?DSN required, e.g. postgresql://vino:pass@127.0.0.1:5432/vino}"

psql "$DSN" -v ON_ERROR_STOP=1 -f "$ROOT/sql/init_pgvector.sql"

if [[ -f "$ROOT/sql/vino_catalog.dump.pgc" ]]; then
  echo "Restoring custom dump vino_catalog.dump.pgc ..."
  pg_restore --no-owner --no-acl --clean --if-exists -d "$DSN" "$ROOT/sql/vino_catalog.dump.pgc" || true
  # --clean may fail on missing; fallback plain:
elif [[ -f "$ROOT/sql/vino_catalog.dump.sql.gz" ]]; then
  echo "Restoring gzipped SQL ..."
  gzip -dc "$ROOT/sql/vino_catalog.dump.sql.gz" | psql "$DSN" -v ON_ERROR_STOP=1
elif [[ -f "$ROOT/sql/vino_catalog.dump.sql" ]]; then
  echo "Restoring plain SQL ..."
  psql "$DSN" -v ON_ERROR_STOP=1 -f "$ROOT/sql/vino_catalog.dump.sql"
else
  echo "No dump found in $ROOT/sql" >&2
  exit 1
fi

echo "OK. Verify: SELECT count(*) FROM wines;  (expect ~2103)"
psql "$DSN" -c "SELECT 'wines' AS t, count(*) FROM wines UNION ALL SELECT 'wineries', count(*) FROM wineries UNION ALL SELECT 'embeddings_siglip2', count(*) FROM embeddings_siglip2 UNION ALL SELECT 'search_photos', count(*) FROM search_photos;"
