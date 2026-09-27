# SQL dumps for server bootstrap

| File | Contents |
|------|----------|
| `init_pgvector.sql` | `CREATE EXTENSION vector` (+ pg_trgm) |
| `vino_catalog.dump.pgc` | pg_dump custom format — **all tables**, **no rows** in `search_photos` / `search_photo_embeddings` |
| `vino_catalog.dump.sql.gz` | same, gzipped plain SQL |

Restore: `../scripts/restore-db.sh 'postgresql://vino:pass@127.0.0.1:5432/vino'`

Do not commit the uncompressed `.sql` (gitignored; regenerate with pg_dump if needed).
