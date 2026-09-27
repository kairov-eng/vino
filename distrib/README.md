# =============================================================================
# Развёртывание Vino Svoe на vino-svoe.online
# Соответствует требованиям РСХБ.Цифра 2026 (см. docs/contest/)
# =============================================================================

## Архитектура на сервере

| Компонент | Как |
|-----------|-----|
| PostgreSQL + **pgvector** | Уже установленный Postgres на хосте (не в Docker) |
| `vino_backend` | Docker: FastAPI findwine `:8092` |
| `vino_frontend` | Docker: nginx + React build `:80` (host `8091`) |
| SigLIP / DINOv3 / CE | Уже существующие контейнеры за nginx (`/api_siglip2` …) |
| Media + веса YOLO/XGB | Каталоги на диске `/var/lib/vino-svoe/...` (volume) |

Контейнеры **обязательно** с префиксом `vino_`: `vino_backend`, `vino_frontend`.

```
Internet → nginx (vino-svoe.online, TLS)
              ├─ /              → vino_frontend
              ├─ /api/ /media/  → vino_backend (или через frontend proxy)
              ├─ /api_siglip2/  → siglip2-embed (как сейчас)
              └─ /api_dinov3/   → dinov3-embed
                     │
              vino_backend ──► Postgres(host) + /var/lib/vino-svoe/media
```

## 0. Предварительные условия

- Docker + Docker Compose v2
- Доступ `ssh aidispatcher` (или ваш хост)
- Сеть Docker `aidispatcher_aidnet` (или поправьте `AIDNET_NAME` в `.env`)
- DNS `vino-svoe.online` → сервер, сертификат Let’s Encrypt (уже есть скрипты в `services/server-nginx/`)

## 1. Подготовка Postgres (без нового контейнера БД)

```bash
sudo -u postgres psql <<'SQL'
CREATE USER vino WITH PASSWORD 'STRONG_PASSWORD';
CREATE DATABASE vino OWNER vino;
\c vino
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
GRANT ALL ON SCHEMA public TO vino;
SQL
```

Или используйте готовый файл:

```bash
psql -U postgres -d vino -f distrib/sql/init_pgvector.sql
```

## 2. Восстановление каталога (дамп с DEV)

Дамп в `distrib/sql/` содержит **все таблицы**, данные по всем **кроме**  
`search_photos` и `search_photo_embeddings` (схема таблиц пустая — для истории сканов).

```bash
chmod +x distrib/scripts/*.sh
export PGPASSWORD='STRONG_PASSWORD'
./distrib/scripts/restore-db.sh "postgresql://vino:STRONG_PASSWORD@127.0.0.1:5432/vino"
```

Проверка: `wines` ≈ 2103, `search_photos` = 0.

## 3. Медиа и модели на диск (не в образ)

```bash
sudo mkdir -p /var/lib/vino-svoe/{media,search_photos,models/yolo,models/yolo_bottle_label,models/xgboost_text_matcher_abs_v14,hf-cache}
```

Скопируйте с рабочей машины (пример):

```bash
# каталог этикеток (crop/label/...)
rsync -avP ./media/uploads/ aidispatcher:/var/lib/vino-svoe/media/

# веса
rsync -avP ./backend/models/yolo/*.pt aidispatcher:/var/lib/vino-svoe/models/yolo/
rsync -avP ./backend/models/yolo_bottle_label/best.pt aidispatcher:/var/lib/vino-svoe/models/yolo_bottle_label/
rsync -avP ./backend/models/xgboost_text_matcher_abs_v14/ aidispatcher:/var/lib/vino-svoe/models/xgboost_text_matcher_abs_v14/
```

В git **нет** фото и `.pt` — только инструкции.

## 4. Секреты

```bash
cp distrib/.env.example distrib/.env
# заполните DATABASE_URL, EMBED_API_KEY, OCR/LLM ключи
```

`DATABASE_URL` должен указывать на Postgres **хоста** с точки зрения контейнера  
(часто IP шлюза docker0 / `host.docker.internal` / IP aidnet).

## 5. Сборка и запуск

Из корня репозитория `vino-svoe`:

```bash
chmod +x distrib/scripts/deploy.sh
./distrib/scripts/deploy.sh
# или:
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d --build
```

Проверка:

```bash
curl -s http://127.0.0.1:8092/api/health
curl -sI http://127.0.0.1:8091/
```

## 6. Nginx публичного домена

1. Подключите фрагмент `distrib/nginx/vino-svoe.app.conf.example` к vhost  
   (или обновите `services/server-nginx/vino-svoe.https.conf.template`).
2. Пересоберите/перезагрузите nginx aidispatcher (`apply-nginx-config.sh`).
3. Убедитесь, что `vino_frontend` и `vino_backend` в сети `aidispatcher_aidnet`.

Открыть: https://vino-svoe.online/

## 7. Eval организатора (приёмка)

```bash
# контракт: POST multipart → {"slug":"..."}
# см. eval/ и architecture/API.md
./eval/participant_test.sh \
  --images-dir eval/queries \
  --manifest eval/queries.tsv \
  --endpoint https://vino-svoe.online/api/eval/predict \
  --output /tmp/predictions.jsonl
```

(Если endpoint ещё называется иначе — см. `docs/contest/ACCEPTANCE.md`.)

## 8. Обновление

```bash
git pull
docker compose -f distrib/docker-compose.yml --env-file distrib/.env up -d --build
```

Медиа и модели на диске **не** пересобираются.

## 9. Обучение моделей (не в Docker-образе)

- **SigLIP2 (Colab):** [`../training/siglip2/README.md`](../training/siglip2/README.md)  
  1) кропы: monorepo `research/label_detect/2.crop_wines.py` → `media/uploads/crop/*_crop.png`  
  2) датасет: `training/siglip2/prepare_siglip2_finetune_dataset.py` → zip на Drive  
  3) Colab: `training/siglip2/colab_train_siglip2.py` (GPU) → веса в Drive / HF Endpoint  
  Веса SigLIP2 **не** в git.
- **XGBoost:** архив [`../training/xgboost/xgboost_train_bundle.zip`](../training/xgboost/xgboost_train_bundle.zip)  
  (скрипты `ocr_matcher/` + parquet `data_xgboost_id800_abs_v14/`) + README;  
  прод-модель **в git:** `backend/models/xgboost_text_matcher_abs_v14/` (`XGB_MODEL_DIR`)
- **Cross-Encoder / SigLIP веса** в git не кладём (`*.safetensors`) — только rsync на сервер
