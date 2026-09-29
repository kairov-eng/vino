# =============================================================================
# Развёртывание Vino Svoe на vino-svoe.online
# Канон `distrib/` приведён к тому, что реально крутится на aidispatcher.
# =============================================================================

## Карта на сервере (факт)

| Что | Где на диске | Контейнер | Порт на хосте |
|-----|--------------|-----------|---------------|
| Код + compose приложения | `/opt/vino-svoe` | — | — |
| Postgres + pgvector | volume `/var/lib/vino-svoe/postgres` | `vino_postgres` | `127.0.0.1:5433` |
| FastAPI findwine | image из `distrib/backend` | `vino_backend` | `127.0.0.1:8092` |
| React + nginx static | image из `distrib/frontend` | `vino_frontend` | `127.0.0.1:8088` |
| SigLIP2 embed | **`/opt/siglip2`** (канон в git: `services/siglip-server` + этот `docker-compose.embeddings.yml`) | `siglip2-embed` | `127.0.0.1:8090` |
| DINOv3 embed | **`/opt/dinov3`** | `dinov3-embed` | `127.0.0.1:8091` |
| Gemini Vision Proxy | **`/opt/gemini-vision-proxy`** (зеркало: `distrib/gemini-vision-proxy/`) | `vino-gemini-vision-proxy` | `8093` |
| Cross-Encoder | `/opt/vino-svoe/services/cross-encoder-server` | `cross-encoder-matcher` | `127.0.0.1:8094` |
| Публичный nginx TLS | `/opt/aidispatcher/distrib/nginx/` | `aidispatcher-nginx` | 80/443 |
| Медиа / модели / секреты | `/var/lib/vino-svoe/{media,media_thumbs,models,secrets,search_photos,hf-cache}` | volume mounts | — |

`/media/` на публичном nginx отдаётся **напрямую с диска** (`/var/www/vino-media`), без прокси в FastAPI.  
Превью сетки: `/media/t/...` → `/var/lib/vino-svoe/media_thumbs` (скрипт `distrib/scripts/build_media_thumbs.py`).

Сеть Docker для nginx↔контейнеры: **`aidispatcher_aidnet`**.  
Сеть приложения: **`vino_net`** (postgres ↔ backend).

```
Internet → aidispatcher-nginx (vino-svoe.online, TLS)
              ├─ /                      → vino_frontend:80
              ├─ /api/                  → vino_backend:8092
              ├─ /media/                → vino_frontend:80  (static, Cache-Control)
              ├─ /api_siglip2/          → siglip2-embed:8090
              ├─ /api_dinov3/           → dinov3-embed:8091
              ├─ /api_cross_encoder_matcher/ → cross-encoder-matcher:8094
              ├─ /health , /v1/         → vino-gemini-vision-proxy:8093
              └─ /v1/eval/predict       → 404 (публичный alias: /v1/eval/predict_public)
                     │
              vino_backend ──► vino_postgres (pgvector)
                            ──► /var/lib/vino-svoe/media + models
```

**Не** использовать Postgres aidispatcher (alpine без `vector`). Каталог Vino — только `vino_postgres`.

Шаблон nginx: [`nginx/vino-svoe.https.conf.template`](nginx/vino-svoe.https.conf.template)  
(= `services/server-nginx/…`, на сервере копируется в `/opt/aidispatcher/distrib/nginx/`).

---

## Стек приложения (`vino_*`)

```bash
# на сервере из /opt/vino-svoe
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d --build
# или
./distrib/scripts/deploy.sh
```

Файлы: [`docker-compose.yml`](docker-compose.yml), [`.env.example`](.env.example).

Проверка:

```bash
curl -fsS http://127.0.0.1:8092/api/health
curl -fsS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8088/
```

---

## Эмбеддинги SigLIP2 / DINOv3

**Как сейчас на проде:** отдельные compose в `/opt/siglip2` и `/opt/dinov3`  
(named volume для HF cache, сеть `aidispatcher_aidnet`).

**Как в репозитории (тот же результат):**

```bash
cp distrib/embeddings.env.example distrib/embeddings.env   # EMBED_API_KEY = как в backend
docker compose -f distrib/docker-compose.embeddings.yml \
  --env-file distrib/embeddings.env up -d --build
```

После смены env у ML — **recreate**, не `restart`:

```bash
docker compose -f distrib/docker-compose.embeddings.yml --env-file distrib/embeddings.env \
  up -d --force-recreate --no-deps siglip2 dinov3
```

Исходники образов: `services/siglip-server/`, `services/dinov3-server/`.

В backend `.env`:

```env
SIGLIP2_ENDPOINT=https://vino-svoe.online/api_siglip2
DINOV3_ENDPOINT=https://vino-svoe.online/api_dinov3
EMBED_API_KEY=<тот же, что у embed-контейнеров>
```

---

## Gemini Vision Proxy

```bash
cp distrib/gemini-proxy.env.example distrib/gemini-proxy.env
docker compose -f distrib/docker-compose.gemini-proxy.yml \
  --env-file distrib/gemini-proxy.env up -d --build
```

Прод-каталог: `/opt/gemini-vision-proxy` (исходники зеркалятся в `distrib/gemini-vision-proxy/`).  
Backend: `GEMINI_PROXY_URL=https://vino-svoe.online`.

---

## Cross-Encoder

Отдельный стек: `services/cross-encoder-server/`  
На сервере: контейнер `cross-encoder-matcher` (`127.0.0.1:8094`), path `/api_cross_encoder_matcher/`.

---

## 1. Postgres (`vino_postgres`)

```bash
sudo mkdir -p /var/lib/vino-svoe/postgres
```

В `distrib/.env`:

```env
VINO_PG_USER=vino
VINO_PG_PASSWORD=STRONG_PASSWORD
VINO_PG_DB=vino
VINO_PGDATA_HOST=/var/lib/vino-svoe/postgres
VINO_PG_PORT=127.0.0.1:5433
DATABASE_URL=postgresql+psycopg2://vino:STRONG_PASSWORD@vino_postgres:5432/vino
```

```bash
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d vino_postgres
docker exec -i vino_postgres psql -U vino -d vino -v ON_ERROR_STOP=1 < distrib/sql/init_pgvector.sql
```

Проверка с хоста: `psql -h 127.0.0.1 -p 5433 -U vino -d vino`.

## 2. Восстановление каталога (дамп)

Дамп в `distrib/sql/` — все таблицы; `search_photos` / `search_photo_embeddings` пустые по данным.

```bash
chmod +x distrib/scripts/*.sh
export PGPASSWORD='STRONG_PASSWORD'
./distrib/scripts/restore-db.sh "postgresql://vino:STRONG_PASSWORD@127.0.0.1:5433/vino"
```

Проверка: `wines` ≈ 2103, `search_photos` = 0.

## 3. Медиа и модели на диск (не в образ)

```bash
sudo mkdir -p /var/lib/vino-svoe/{media,search_photos,models/yolo,models/yolo_bottle_label,models/xgboost_text_matcher_abs_v14,hf-cache,secrets}
```

```bash
rsync -avP ./media/uploads/ aidispatcher:/var/lib/vino-svoe/media/
rsync -avP ./backend/models/yolo/*.pt aidispatcher:/var/lib/vino-svoe/models/yolo/
rsync -avP ./backend/models/yolo_bottle_label/best.pt aidispatcher:/var/lib/vino-svoe/models/yolo_bottle_label/
rsync -avP ./backend/models/xgboost_text_matcher_abs_v14/ aidispatcher:/var/lib/vino-svoe/models/xgboost_text_matcher_abs_v14/
```

В git **нет** фото и YOLO `.pt`.

## 4. Секреты

```bash
cp distrib/.env.example distrib/.env
# пароль PG, EMBED_API_KEY, OCR/LLM, password_*
```

`DATABASE_URL` внутри compose: хост **`vino_postgres`**, порт **5432**.

## 5. Nginx публичного домена

1. Скопировать [`nginx/vino-svoe.https.conf.template`](nginx/vino-svoe.https.conf.template)  
   → `/opt/aidispatcher/distrib/nginx/` (или через `services/server-nginx/apply-nginx-config.sh`).
2. Все перечисленные контейнеры в `aidispatcher_aidnet`.
3. `nginx -t` + reload.

Сайт: https://vino-svoe.online/

## 6. Eval организатора

Публичный путь на проде (см. nginx):

```bash
./eval/participant_test.sh \
  --images-dir eval/queries \
  --manifest eval/queries.tsv \
  --endpoint https://vino-svoe.online/v1/eval/predict_public \
  --output /tmp/predictions.jsonl
```

(`/v1/eval/predict` без `_public` на nginx отдаёт 404.)

## 7. Обновление

```bash
# приложение
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d --build

# embeddings (если меняли код/env) — на проде чаще:
#   cd /opt/siglip2 && docker compose up -d --build --force-recreate
#   cd /opt/dinov3  && docker compose up -d --build --force-recreate
# или из репо:
docker compose -f distrib/docker-compose.embeddings.yml --env-file distrib/embeddings.env \
  up -d --build --force-recreate
```

Медиа и `.pt` на диске **не** пересобираются.

## 8. Обучение моделей (не в Docker-образе backend)

- **SigLIP2 (Colab):** [`../training/siglip2/README.md`](../training/siglip2/README.md)
- **YOLO / XGBoost данные:** [`../Подготовка данных/`](../Подготовка%20данных/)
- **XGBoost train-bundle:** [`../training/xgboost/`](../training/xgboost/)  
  прод-модель: `backend/models/xgboost_text_matcher_abs_v14/`
