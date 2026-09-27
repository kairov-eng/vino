# API backend (Vino Svoe)

Базовый URL:

| Среда | URL |
|-------|-----|
| Dev | `http://127.0.0.1:8092` |
| Prod | `https://vino-svoe.online` (nginx → `vino_backend` / `vino_frontend`) |

Префикс приложения: **`/api`**. Eval организатора: **`/v1/eval/predict`** (без `/api`).  
Ответы — `application/json; charset=utf-8`, кроме `/media/*` (бинарные файлы).

При включённом site gate (`password_admin` / `password_user` в `.env`) запросы к `/api/*` требуют cookie `vino_site_password` или заголовок `X-Site-Password`.  
Исключения: `GET /api/health`, `GET|POST /api/site-auth`, `OPTIONS`, **`/media/*`**.

---

## POST /v1/eval/predict — контракт организатора

`multipart/form-data`, поле `image` (JPEG/PNG/WEBP/HEIC, ≤ ~20 МБ).

Ответ 200:

```json
{"slug": "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16"}
```

`slug` — лучший кандидат пайплайна (может быть `null`, если победителя нет).  
Скрипт: `eval/participant_test.sh` (curl timeout 10 с).

---

## POST /api/findwine — основной поиск (UI)

`multipart/form-data`, поле `image` (+ опциональные флаги в query/body по схеме).

Ответ (фрагмент):

```json
{
  "search_photos_id": 2743,
  "algorithm_version": "0.97.1",
  "algorithm_updated_at": "2026-09-27T22:10:00+03:00",
  "crops_url": "/api/search-photos/….jpg",
  "label_url": "/api/search-photos/…_label.jpg",
  "candidates_siglip2": [ { "id": 2306, "cosine_similarity": 0.91, "slug": "…", "xgb_score": 0.88, "…": "…" } ],
  "candidates_dinov3": [],
  "matched_wine_id": 2306,
  "matched_wine_confidence": 0.87,
  "matched_wine_slug": "…",
  "matched_wine_photo_url": "/media/….webp",
  "manual_wines_id": null,
  "status": { "steps": { }, "timings_ms": { }, "decision": "…" }
}
```

Полный разбор сигналов — в `status` (OCR, exclusive, FinalScore, reuse и т.д.). См. `docs/findwine-pipeline.md`.

Повторная выдача: `GET /api/findwine/{search_photos_id}`.

---

## Каталог

### GET /api/wines

Список с фильтрами (`q`, `category`, `region`, `winery`, `offset`, `limit` …).

### GET /api/wines/filters

Доступные значения фильтров.

### GET /api/wines/{slug}

Карточка: поля каталога + `photo_url`, `label_url`, опц. `label_ocr`.

---

## Скан / история

| Метод | Путь | Назначение |
|-------|------|------------|
| POST | `/api/scan` | сохранить фото (без полного пайплайна) |
| POST | `/api/preview-image` | превью / служебная обработка кадра |
| GET | `/api/scan-history` | история поисков |
| GET | `/api/scan-history/report` | агрегированный отчёт |
| GET | `/api/search-photos/{filename}` | файл запроса (crops/label) |

---

## Настройки и доступ

| Метод | Путь | Назначение |
|-------|------|------------|
| GET | `/api/settings` | текущие настройки пайплайна |
| PUT/PATCH | `/api/settings` | сохранить (роль **admin**) |
| GET | `/api/site-auth` | проверка cookie/пароля |
| POST | `/api/site-auth` | логин site gate `{ "password": "…" }` → `{ ok, role }` |
| GET | `/api/health` | `{ "status": "ok", "media": { … } }` |

---

## Медиа

`GET /media/{path}` — эталонные фото каталога и `crop/*_crop.png`.

- **Прод:** nginx `vino_frontend` (alias на volume), `Cache-Control: public, max-age=2592000, immutable`.
- **Dev:** FastAPI `FileResponse` из `MEDIA_ROOT`.

Примеры: `/media/01_Riesling_….webp`, `/media/crop/….webp_crop.png`.

---

## Внешние ML (тот же хост, другие контейнеры)

Не часть `vino_backend`, но используются им по URL из `.env`:

| Префикс nginx | Сервис |
|---------------|--------|
| `/api_siglip2/` | siglip2-embed |
| `/api_dinov3/` | dinov3-embed |
| `/api_cross_encoder_matcher/` | cross-encoder-matcher |
| `/v1/` (Gemini proxy) | vino-gemini-vision-proxy |

Ключ эмбеддингов: `EMBED_API_KEY`.
