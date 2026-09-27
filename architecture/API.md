# API ml-service (v1)

Базовый URL: `http://127.0.0.1:8080`. Все ответы — `application/json; charset=utf-8`. Ошибки — `{"error": {"code": "...", "message": "..."}}` с HTTP 4xx/5xx; `/v1/eval/predict` при любых внутренних проблемах стремится вернуть 200 с лучшим доступным slug.

## POST /v1/eval/predict — контракт организатора

Запрос: `multipart/form-data`, поле `image` (JPEG/PNG/WEBP/HEIC, ≤ 20 МБ).

Ответ 200:

```json
{"slug": "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16"}
```

Гарантии: всегда объект с непустой строкой `slug` (top-1 реранкера независимо от `decision`); время ответа ≤ 3 с на целевом железе; никаких лишних полей (скрипт читает `.slug`, но держим ответ плоским).

## POST /v1/scan — расширенный ответ для UI и диагностики

Запрос: `multipart/form-data`, поле `image`; query `debug=0|1`, `topk=5`.

Ответ 200:

```json
{
  "request_id": "01J7Z…",
  "decision": "match",
  "confidence": { "top1": 0.93, "top5": [0.93, 0.31, 0.12, 0.08, 0.05] },
  "margin": 0.62,
  "wine": { "...карточка, как в GET /v1/wines/{slug}..." },
  "top5": [
    { "slug": "…", "name": "…", "winery": "…", "image": "/media/…", "score": 0.91,
      "signals": { "siglip": 0.83, "dino": 0.79, "inliers": 148, "text": 0.88, "year_match": 1 } }
  ],
  "similar": [],
  "timings_ms": { "normalize": 58, "embed": 410, "ocr": 520, "local": 90, "search": 6, "rerank": 340, "total": 1210 },
  "warnings": [],
  "debug": {
    "boxes": { "bottle": [x, y, w, h], "label": [x, y, w, h] },
    "crops": { "label": "/tmp-media/…png", "bottle": "…" },
    "ocr": { "text": "МАССАНДРА … 2023", "year": 2023, "sweetness": null, "volume": 0.75 },
    "candidates": 37
  }
}
```

`decision`: `match` | `low_confidence` | `not_found`. При `not_found` поле `wine` = `null`, `similar` содержит 3–5 аналогов. `debug` присутствует только при `debug=1`.

## GET /v1/wines/{slug}

```json
{
  "slug": "…", "name": "…", "portal_url": "https://vino-svoe.ru/wines/…",
  "winery": { "slug": "…", "name": "…", "cover": "/media/…" },
  "region": { "name": "Крым", "slug": "krym", "has_geojson": true },
  "category": "Белое", "color": "Светло-соломенный", "grapes": ["Алиготе", "Кокур Белый"],
  "sweetness": "сухое", "alcohol": 13.5, "volume": 0.75, "year": null,
  "description": "…",
  "image": "/media/…webp",
  "serving": { "temp_c": [8, 10], "pairing": ["…"] }
}
```

`serving` — детерминированные подсказки по правилам (цвет/сладость/сорт), не LLM.

## GET /v1/wines/{slug}/similar?limit=5&other_wineries=1

```json
{ "items": [ { "slug": "…", "name": "…", "winery": "…", "image": "…", "score": 0.81,
              "why": ["тот же сорт: Рислинг", "тот же регион: Крым", "другая винодельня"] } ] }
```

## GET /v1/regions/{slug}/geojson — полигон региона для карты.

## POST /v1/sommelier

```json
{ "slug": "…", "session_id": "…", "answers": { "occasion": "ужин", "dish": "рыба", "style": "свежее" } }
```

Ответ: либо следующий вопрос `{ "question": { "id": "dish", "text": "…", "options": ["…"] } }`, либо рекомендация
`{ "recommendation": { "slug": "…", "why": "…", "pairing": ["…"], "serving_temp_c": [8,10] }, "alternatives": [ … ], "provider": "openai|fallback" }`.

## POST /v1/feedback

```json
{ "request_id": "01J7Z…", "is_correct": false, "correct_slug": "…" }
```

## GET /v1/health → `{ "status": "ok", "models": {"siglip2": "loaded", …}, "index": { "wines": 2103, "vectors": 41860, "loaded_at": "…" } }`

## GET /v1/metrics — Prometheus text (латентности по этапам, счётчики decision).

## POST /v1/admin/reload — перечитать индекс из БД (после `svoe_vino index`). Защищено `ADMIN_TOKEN`.

## Медиа

`GET /media/{path}` — статика эталонных изображений (только чтение, из `MEDIA_DIR`).
