# Установка и настройка Vino Svoe

Документ для корня монорепозитория `C:\dev\Vino2026\`.  
Простым языком: что читать, какие ключи нужны, в каком порядке поднимать систему.

---

## Доступ к демо-сайту (оценка)

Сайт выложен в свободный доступ, поэтому **на период оценки он запаролен**.

- Адрес: https://vino-svoe.online  
- Пароль пользователя: **`lct2026_user`**

Введите его на экране входа (gate). После входа открываются сканер, история и каталог.  
(Админский пароль — отдельный, для настроек пайплайна; участникам оценки нужен пользовательский.)

---

## Какие документы зачем

| Файл | Что внутри |
|------|------------|
| [`vino-svoe/README.md`](vino-svoe/README.md) | Продуктовый README: быстрый старт front+back, таблица документации, что в git / что нет, кратко про прод |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) (копия) / [`vino-svoe/architecture/ARCHITECTURE.md`](vino-svoe/architecture/ARCHITECTURE.md) | Архитектура: схема сервисов, пайплайн findwine, таблицы БД, API-сводка, ADR, деплой |
| [`STRUCTURE.md`](STRUCTURE.md) | Карта папок на диске (`vino-svoe`, `media`, `research`, …) |
| [`vino-svoe/distrib/README.md`](vino-svoe/distrib/README.md) | Как развернуть на сервере (Docker, media, модели) |
| [`vino-svoe/docs/findwine-pipeline.md`](vino-svoe/docs/findwine-pipeline.md) | Подробно, как работает поиск по фото |
| [`vino-svoe/Подготовка данных/`](vino-svoe/Подготовка%20данных/) | Скрипты crop / YOLO (Colab) / XGBoost |
| **Этот файл (`УСТАНОВКА.md`)** | Последовательность установки и список ключей / endpoint |

Канон архитектуры живёт в `vino-svoe/architecture/`. Копия в корне — чтобы её было видно рядом со `STRUCTURE.md` без захода в подпапку.

---

## Что нужно для запуска (ключи и endpoint)

Секреты кладутся в `vino-svoe/backend/.env` (образец: `backend/.env.example`). **Не коммитить** реальные ключи.

### Обязательный минимум (локальный поиск «заведётся»)

| Переменная | Зачем |
|------------|--------|
| `DATABASE_URL` | Postgres **в контейнере** `vino_postgres` (pgvector); в compose: `…@vino_postgres:5432/vino` |
| `MEDIA_ROOT` | Папка фото каталога (локально часто `C:\dev\Vino2026\media\uploads`) |
| `SIGLIP2_ENDPOINT` | HTTP SigLIP2 — прод: `https://vino-svoe.online/api_siglip2` (контейнер `siglip2-embed`) |
| Веса YOLO | пути `YOLO_MODEL_PATH` / `YOLO_BOTTLE_LABEL_MODEL_PATH` к `.pt` |
| `XGB_MODEL_DIR` | каталог с `model.json` текстового матчера |

Без внешних OCR (только RapidOCR) и без GPU Hugging Face система уже умеет искать, но качество OCR/скорость embeddings будут скромнее.

### Hugging Face (опционально, GPU-эмбеддинги / Qwen OCR)

Нужны, если в настройках сканера выбран режим **GPU** для embeddings или включён Qwen OCR.

| Переменная | Смысл |
|------------|--------|
| `HF_ACCESS_TOKEN` | токен HF с правом вызова Inference Endpoints |
| `HF_SIGLIP2_ENDPOINT` | URL Inference Endpoint SigLIP2 (или общий `HF_ENDPOINT`) |
| `HF_DINOV3_ENDPOINT` | URL Endpoint DINOv3 (если используете) |
| `HF_NAMESPACE` + `HF_SIGLIP2_ENDPOINT_NAME` / `HF_DINOV3_ENDPOINT_NAME` | для авто-start «уснувшего» endpoint |
| `HF_START_TIMEOUT_SEC` | сколько ждать прогрева (дальше — fallback на свой CPU endpoint) |
| `HF_QWEN_OCR_ENDPOINT` (+ `_NAME`, `_MODEL`) | если включён Qwen OCR |

Типичный сценарий: GPU HF → при паузе/ошибке backend уходит на `SIGLIP2_ENDPOINT` / `DINOV3_ENDPOINT` (CPU-контейнеры на сервере).

### Google Vision OCR

| Что | Как |
|-----|-----|
| Включение | в UI настроек сканера и/или флаг `USE_GOOGLE_VISION_OCR=1` |
| Авторизация | файл service account JSON → `GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\sa.json` |
| Фича | `GOOGLE_VISION_FEATURE=DOCUMENT_TEXT_DETECTION` (по умолчанию) |
| Облачный проект | у SA должны быть права на Cloud Vision API |

Отдельного «API key строки» для Vision в `.env` обычно нет — используется Google Cloud credentials JSON.

### Другие полезные ключи (по желанию)

| Сервис | Переменные |
|--------|------------|
| Gemini OCR / crop | `GEMINI_API_KEY` или `GEMINI_API_KEY_1…`; опц. `GEMINI_PROXY_URL` |
| OpenAI OCR / txt-match | `OPENAI_KEY` |
| DeepSeek OCR | `DEEPSEEK_API_KEY` |
| Yandex OCR | `YANDEX_API_KEY`, `YANDEX_FOLDER_ID` |
| Site gate (пароль сайта) | `password_user`, `password_admin` |
| Embed API (если сервисы защищены) | `EMBED_API_KEY` |

На проде пароль оценки: `password_user=lct2026_user` (см. выше).

### Endpoint’ы ML на проде (тот же сервер, контейнеры)

| Путь | Контейнер | Compose |
|------|-----------|---------|
| `/api_siglip2` | `siglip2-embed` (:8090) | `distrib/docker-compose.embeddings.yml` |
| `/api_dinov3` | `dinov3-embed` (:8091) | то же |
| `/api_cross_encoder_matcher` | Cross-Encoder | `services/cross-encoder-server/` |
| `/v1/` (proxy) | Gemini vision proxy | отдельно |

Локально в `.env` можно указать URL `vino-svoe.online` **или** `http://127.0.0.1:8090` / `:8091` после подъёма embeddings compose.

---

## Последовательность установки (dev)

### 1. Код и окружение

1. Клонировать / открыть репозиторий продукта: папка `vino-svoe/` (GitHub: `kairov-eng/vino`).
2. Установить **Python 3.11+**, **Node.js 20+**, **Docker** (для прод-стека; локально Postgres можно также через `vino_postgres` из compose).
3. Прод/сервер: поднять `vino_postgres` из `distrib/docker-compose.yml` (образ `pgvector/pgvector:pg17`) — **не** хостовый Postgres aidispatcher.

### 2. Backend

```powershell
cd C:\dev\Vino2026\vino-svoe\backend
copy .env.example .env
# заполните DATABASE_URL, MEDIA_ROOT, ключи/endpoint из таблицы выше
pip install -r requirements.txt
alembic upgrade head
# при необходимости восстановите дамп каталога из distrib/sql/
python run_dev.py
```

Backend: http://127.0.0.1:8092

### 3. Frontend

```powershell
cd C:\dev\Vino2026\vino-svoe\frontend
npm install
npm run dev
```

UI: http://127.0.0.1:8091  
Vite проксирует `/api` и `/media` на backend.

### 4. Данные и модели

1. Фото каталога → `MEDIA_ROOT` (часто `C:\dev\Vino2026\media\uploads`, плюс подпапка `crop/`).
2. Веса YOLO `.pt` → пути из `.env`.
3. XGBoost → `XGB_MODEL_DIR` (в git есть `backend/models/xgboost_text_matcher_abs_v14/`).
4. SigLIP2: либо живой `SIGLIP2_ENDPOINT`, либо локальный fallback (`SIGLIP2_LOCAL_*`).

Подготовка новых данных (crop / YOLO Colab / XGB):  
[`vino-svoe/Подготовка данных/`](vino-svoe/Подготовка%20данных/).

### 5. Пароль сайта (опционально локально)

В `backend/.env`:

```env
password_user=lct2026_user
password_admin=...
```

После перезагрузки backend откроется форма входа, как на демо.

### 6. Проверка

1. Открыть http://127.0.0.1:8091 (ввести пароль, если задан).
2. Загрузить фото этикетки → «Поиск».
3. Health: http://127.0.0.1:8092/api/health

---

## Прод (очень кратко)

Полная инструкция: [`vino-svoe/distrib/README.md`](vino-svoe/distrib/README.md).

1. На сервере: Docker Compose **`vino_postgres`** + `vino_backend` + `vino_frontend` (`distrib/docker-compose.yml`).
2. Эмбеддинги на том же хосте: `distrib/docker-compose.embeddings.yml` (`siglip2-embed`, `dinov3-embed`).
3. Media и модели — volume `/var/lib/vino-svoe/...`.
4. В `.env` / `embeddings.env` — ключи HF / Google Vision / Gemini / `EMBED_API_KEY` / site passwords.
5. Демо для оценки: https://vino-svoe.online · пароль **`lct2026_user`**.

---

## Частые вопросы

**Нужен ли сразу Hugging Face?**  
Нет. Для демо достаточно CPU-endpoint SigLIP2. HF ускоряет embeddings на GPU.

**Нужен ли Google Vision?**  
Нет для старта. Включается в настройках; нужен JSON service account.

**Где архитектура «как устроено»?**  
[`ARCHITECTURE.md`](ARCHITECTURE.md).

**Где «как запускать код каждый день»?**  
[`vino-svoe/README.md`](vino-svoe/README.md).
