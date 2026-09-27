# Своё Вино — Сканер вина

Сервис распознавания российского вина по фотографии этикетки с выдачей карточки из каталога платформы «Своё Вино». Конкурсное задание РСХБ.Цифра 2026.

Статус: **проектирование завершено, реализация — этап 1** (см. `ROADMAP.md`).

## Документы

| Документ | Содержание |
|----------|-----------|
| [TZ.md](TZ.md) | Техническое задание: цели, сценарии, функциональные и нефункциональные требования, метрики, риски, приёмка |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Архитектура: пайплайн, границы слоёв, структура репозитория, модель данных, развёртывание |
| [TECH_CHOICE.md](TECH_CHOICE.md) | Выбор технологий для векторов по фото: сравнение моделей, итоговый стек, план экспериментов |
| [DATA.md](DATA.md) | Анализ исходных данных и правила сопоставления slug ↔ фото |
| [API.md](API.md) | Контракты REST API, включая точку для скрипта организатора |
| [ROADMAP.md](ROADMAP.md) | Три этапа реализации с задачами и критериями приёмки |

**Живой пайплайн реализации (findwine, актуальные правила детекции/скоринга):**

| Документ | Содержание |
|----------|-----------|
| [`../docs/findwine-pipeline.md`](../docs/findwine-pipeline.md) | YOLO / bottle+label, exclusive lexicon, XGB, Cross Encoder, `matched_wine` |
| [`../docs/algorithm-ocr-text-finalscore-f1.md`](../docs/algorithm-ocr-text-finalscore-f1.md) | TextScore, FinalScore, F1, hard mismatch |

---

## Кратко об устройстве

Целевая схема конкурса (ниже) частично расходится с текущим `POST /api/findwine` — см. docs выше.
```
фото → нормализация (детекция бутылки/этикетки, кроп) → SigLIP 2 + DINOv2 эмбеддинги ∥ OCR ∥ локальные признаки
     → кандидаты (pgvector + текст) → реранк (геометрия этикетки, атрибуты, OCR) → confidence/margin → карточка JSON
```

- `services/ml` — Python 3.12 / FastAPI / PyTorch: пайплайн, индексатор, API (`:8080`).
- `services/web` — Nuxt 3, mobile-first UI в стилистике портала (`:3000`).
- PostgreSQL 16 + pgvector — каталог, векторы, логи.
- Работает на CPU; GPU ускоряет через `DEVICE=cuda`.

## Запуск (целевой, после этапа 1)

```bash
cp .env.example .env            # пути к CSV и uploads, DEVICE, LLM-провайдер
make setup                      # веса моделей
make ingest index               # каталог → БД → векторы
docker compose up               # db + ml + web
./eval/participant_test.sh --images-dir eval/queries --manifest eval/queries.tsv \
  --endpoint http://127.0.0.1:8080/v1/eval/predict --output predictions.jsonl
```

## Переменные окружения (планируемые)

| Переменная | Назначение | По умолчанию |
|-----------|------------|--------------|
| `DATABASE_URL` | PostgreSQL с расширением `vector` | `postgresql://svoe:svoe@db:5432/svoe` |
| `MEDIA_DIR` | папка `uploads/` Strapi | `./data/media` |
| `CATALOG_CSV` | экспорт каталога | `./data/raw/strapi_output.csv` |
| `DEVICE` | `cpu` \| `cuda` | `cpu` |
| `SIGLIP_MODEL` | `google/siglip2-base-patch16-384` \| `…so400m-patch16-naflex` | base |
| `OCR_ENABLED`, `OCR_TIMEOUT_MS` | текстовый канал | `1`, `700` |
| `LLM_PROVIDER`, `LLM_API_KEY` | `openai` \| `gemini` \| `yandexgpt` \| `ollama` \| `none` | `none` |
| `ADMIN_TOKEN` | защита `/v1/admin/*` | — |

## Ограничения

- 9 из 2103 позиций каталога не имеют эталонного фото — ищутся только по тексту.
- 14 эталонных фото принадлежат двум позициям одновременно — различаются только по OCR.
- Публичный набор полевых фото — 3 снимка; для настройки порогов требуется собственная съёмка (см. `DATA.md` §6).
