-- Run once on the existing Postgres server as superuser:
--   psql -U postgres -d vino -f init_pgvector.sql
-- Or create DB first:
--   CREATE USER vino WITH PASSWORD '...';
--   CREATE DATABASE vino OWNER vino;
--   \c vino

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Optional: grant to app role
-- GRANT ALL ON SCHEMA public TO vino;
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO vino;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO vino;
