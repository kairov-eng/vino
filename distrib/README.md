# =============================================================================
# Развёртывание Vino Svoe на vino-svoe.online
# Соответствует требованиям РСХБ.Цифра 2026 (см. docs/contest/)
# =============================================================================

## Архитектура на сервере

| Компонент | Как |
|-----------|-----|
| PostgreSQL + **pgvector** | Контейнер **`vino_postgres`** (`pgvector/pgvector:pg17`), данные в `/var/lib/vino-svoe/postgres` |
| `vino_backend` | Docker: FastAPI findwine `:8092` |
| `vino_frontend` | Docker: nginx + React build (host probe `:8088`) |
| SigLIP2 / DINOv3 | Контейнеры **`siglip2-embed`** / **`dinov3-embed`** — [`docker-compose.embeddings.yml`](docker-compose.embeddings.yml) |
| Cross-Encoder / Gemini proxy | Отдельные контейнеры за nginx (`/api_cross_encoder_matcher`, `/v1/`) |
| Media + веса YOLO/XGB | Каталоги на диске `/var/lib/vino-svoe/...` (volume) |

Стек приложения (`vino_*`):

```bash
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d --build
```

Эмбеддинги на том же сервере:

```bash
cp distrib/embeddings.env.example distrib/embeddings.env   # заполнить EMBED_API_KEY
docker compose -f distrib/docker-compose.embeddings.yml --env-file distrib/embeddings.env up -d --build
```

```
Internet → nginx (vino-svoe.online, TLS)
              ├─ /                 → vino_frontend
              ├─ /api/             → vino_backend
              ├─ /media/           → static (vino_frontend volume)
              ├─ /api_siglip2/     → siglip2-embed:8090
              └─ /api_dinov3/      → dinov3-embed:8091
                     │
              vino_backend ──► vino_postgres (pgvector) + /var/lib/vino-svoe/media
```

**Не** использовать Postgres aidispatcher (alpine без `vector`). Каталог Vino — только `vino_postgres`.

## 0. Предварительные условия

- Docker + Docker Compose v2
- Доступ `ssh aidispatcher` (или ваш хост)
- Сеть Docker `aidispatcher_aidnet` (или поправьте `AIDNET_NAME` в `.env`)
- DNS `vino-svoe.online` → сервер, сертификат Let’s Encrypt (`services/server-nginx/`)

## 1. Postgres в контейнере (`vino_postgres`)

Поднимается вместе с backend из `docker-compose.yml`. Данные на хосте:

```bash
sudo mkdir -p /var/lib/vino-svoe/postgres
```

В `distrib/.env`:

```env
VINO_PG_USER=vino
VINO_PG_PASSWORD=STRONG_PASSWORD
VINO_PG_DB=vino
VINO_PGDATA_HOST=/var/lib/vino-svoe/postgres
# host-only порт для psql / restore с хоста
VINO_PG_PORT=127.0.0.1:5433
# URL для backend (имя сервиса compose, не localhost)
DATABASE_URL=postgresql+psycopg2://vino:STRONG_PASSWORD@vino_postgres:5432/vino
```

Старт только БД:

```bash
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d vino_postgres
```

Расширения (после healthy):

```bash
docker exec -i vino_postgres psql -U vino -d vino -v ON_ERROR_STOP=1 < distrib/sql/init_pgvector.sql
```

Проверка с хоста: `psql -h 127.0.0.1 -p 5433 -U vino -d vino`.

## 2. Восстановление каталога (дамп)

Дамп в `distrib/sql/` — все таблицы; данные по всем **кроме**  
`search_photos` / `search_photo_embeddings` (схема пустая).

```bash
chmod +x distrib/scripts/*.sh
export PGPASSWORD='STRONG_PASSWORD'
./distrib/scripts/restore-db.sh "postgresql://vino:STRONG_PASSWORD@127.0.0.1:5433/vino"
# или server_restore_and_build.sh на сервере
```

Проверка: `wines` ≈ 2103, `search_photos` = 0.

## 3. Медиа и модели на диск (не в образ)

```bash
sudo mkdir -p /var/lib/vino-svoe/{media,search_photos,models/yolo,models/yolo_bottle_label,models/xgboost_text_matcher_abs_v14,hf-cache/siglip2,hf-cache/dinov3}
```

```bash
rsync -avP ./media/uploads/ aidispatcher:/var/lib/vino-svoe/media/
rsync -avP ./backend/models/yolo/*.pt aidispatcher:/var/lib/vino-svoe/models/yolo/
rsync -avP ./backend/models/yolo_bottle_label/best.pt aidispatcher:/var/lib/vino-svoe/models/yolo_bottle_label/
rsync -avP ./backend/models/xgboost_text_matcher_abs_v14/ aidispatcher:/var/lib/vino-svoe/models/xgboost_text_matcher_abs_v14/
```

В git **нет** фото и `.pt` — только инструкции.

## 4. Секреты приложения

```bash
cp distrib/.env.example distrib/.env
# VINO_PG_PASSWORD, DATABASE_URL → @vino_postgres, EMBED_API_KEY, OCR/LLM, password_*
```

`DATABASE_URL` внутри сети compose: хост **`vino_postgres`**, порт **5432** (не `127.0.0.1` хоста).

## 5. SigLIP2 + DINOv3 (тот же сервер)

Compose: [`docker-compose.embeddings.yml`](docker-compose.embeddings.yml)  
Env-пример: [`embeddings.env.example`](embeddings.env.example)

| Контейнер | Порт в контейнере | Host bind (по умолчанию) | Публичный path |
|-----------|-------------------|--------------------------|----------------|
| `siglip2-embed` | 8090 | `127.0.0.1:8090` | `/api_siglip2/` |
| `dinov3-embed` | 8091 | `127.0.0.1:8091` | `/api_dinov3/` |

```bash
cp distrib/embeddings.env.example distrib/embeddings.env
# тот же EMBED_API_KEY, что в distrib/.env / backend
docker compose -f distrib/docker-compose.embeddings.yml --env-file distrib/embeddings.env up -d --build
curl -fsS http://127.0.0.1:8090/health
curl -fsS http://127.0.0.1:8091/health
```

После смены env у ML-сервисов — **recreate**, не `restart` (см. `.cursor/rules/env-container-recreate.mdc`):

```bash
docker compose -f distrib/docker-compose.embeddings.yml --env-file distrib/embeddings.env \
  up -d --force-recreate --no-deps siglip2 dinov3
```

Исходники образов: `services/siglip-server/`, `services/dinov3-server/` (канон кода).  
`distrib/` держит **прод-compose + env** для того же сервера, что и `vino_*`.

В backend:

```env
SIGLIP2_ENDPOINT=https://vino-svoe.online/api_siglip2
DINOV3_ENDPOINT=https://vino-svoe.online/api_dinov3
EMBED_API_KEY=<тот же, что в embeddings.env>
```

## 6. Сборка и запуск vino_* 

Из корня `vino-svoe`:

```bash
chmod +x distrib/scripts/deploy.sh
./distrib/scripts/deploy.sh
# или:
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d --build
```

Проверка:

```bash
curl -s http://127.0.0.1:8092/api/health
curl -sI http://127.0.0.1:8088/
```

(Host-порт frontend по умолчанию **8088**, чтобы не конфликтовать с `dinov3-embed` на **8091**.)

## 7. Nginx публичного домена

1. Фрагмент `distrib/nginx/vino-svoe.app.conf.example` или `services/server-nginx/vino-svoe.https.conf.template`  
   (`/api_siglip2/` → `siglip2-embed:8090`, `/api_dinov3/` → `dinov3-embed:8091`).
2. Контейнеры embeddings и `vino_*` в сети `aidispatcher_aidnet`.
3. Reload nginx aidispatcher.

Открыть: https://vino-svoe.online/

## 8. Eval организатора

```bash
./eval/participant_test.sh \
  --images-dir eval/queries \
  --manifest eval/queries.tsv \
  --endpoint https://vino-svoe.online/api/eval/predict \
  --output /tmp/predictions.jsonl
```

## 9. Обновление

```bash
# приложение
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d --build

# embeddings (если меняли код/env сервисов)
docker compose -f distrib/docker-compose.embeddings.yml --env-file distrib/embeddings.env up -d --build
```

Медиа и `.pt` на диске **не** пересобираются.

## 10. Обучение моделей (не в Docker-образе backend)

- **SigLIP2 (Colab):** [`../training/siglip2/README.md`](../training/siglip2/README.md)
- **YOLO / XGBoost данные:** [`../Подготовка данных/`](../Подготовка%20данных/)
- **XGBoost train-bundle:** [`../training/xgboost/`](../training/xgboost/)  
  прод-модель в git: `backend/models/xgboost_text_matcher_abs_v14/`
- Веса `*.safetensors` / YOLO `*.pt` в git не кладём (кроме оговорённого XGB)
