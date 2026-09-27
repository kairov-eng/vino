# Vino Svoe — архитектура и ТЗ

Сервис распознавания российского вина по фото этикетки/бутылки с выдачей карточки из каталога «Своё Вино». Конкурс РСХБ.Цифра 2026.

**Статус:** рабочий продукт на https://vino-svoe.online · алгоритм findwine **0.97.x** · стек React + FastAPI + PostgreSQL/pgvector.

## Документы в этой папке

| Документ | Содержание |
|----------|-----------|
| [TZ.md](TZ.md) | Техническое задание: цели, сценарии, требования, метрики, приёмка |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Как устроен сервис сейчас: слои, пайплайн, деплой, данные |
| [TECH_CHOICE.md](TECH_CHOICE.md) | Выбор моделей и технологий (что взяли и почему) |
| [DATA.md](DATA.md) | Исходные данные каталога и правило slug ↔ фото |
| [API.md](API.md) | REST API backend (`/api/*`, eval) |
| [ROADMAP.md](ROADMAP.md) | Этапы: что сделано / что осталось |

## Живые документы реализации (вне этой папки)

| Документ | Содержание |
|----------|-----------|
| [`../docs/findwine-pipeline.md`](../docs/findwine-pipeline.md) | Подробный алгоритм findwine (YOLO, OCR, exclusive, XGB, FinalScore) |
| [`../docs/algorithm-ocr-text-finalscore-f1.md`](../docs/algorithm-ocr-text-finalscore-f1.md) | TextScore / FinalScore / F1 |
| [`../distrib/README.md`](../distrib/README.md) | Деплой на vino-svoe.online |
| [`../docs/contest/`](../docs/contest/) | Приёмка и презентация конкурса |
| [`../training/`](../training/) | Обучение SigLIP2 (Colab) и XGBoost |

## Устройство (кратко)

```
фото → YOLO (этикетка / bottle+label) → кроп
     → SigLIP2 (+ опц. DINOv3) ANN top-20 в pgvector
     → OCR (RapidOCR / Gemini / Google Vision / …) параллельно
     → exclusive lexicon + ColorDelta + XGB / Cross-Encoder
     → FinalScore → matched_wine + аналоги → UI / {"slug"}
```

| Компонент | Реализация |
|-----------|------------|
| UI | React (Vite), dev `:8091`, прод — `vino_frontend` |
| API / пайплайн | FastAPI, dev `:8092`, прод — `vino_backend` |
| БД | PostgreSQL + `vector` + `pg_trgm` (прод: контейнер `vino_postgres`) |
| Эмбеддинги | HTTP: SigLIP2 / DINOv3 / CE (+ опц. HF Inference Endpoints) |
| Медиа | `/media/*` — nginx static с диска (`media/` + `media/crop/`) |

## Быстрый старт (dev)

```bash
# backend
cd backend && cp .env.example .env   # ключи
pip install -r requirements.txt
python run_dev.py                   # http://127.0.0.1:8092

# frontend
cd frontend && npm install && npm run dev   # http://127.0.0.1:8091
```

Eval организатора:

```bash
./eval/participant_test.sh \
  --images-dir eval/queries \
  --manifest eval/queries.tsv \
  --endpoint https://vino-svoe.online/v1/eval/predict \
  --output /tmp/predictions.jsonl
```

## Ограничения каталога

- ~2103 вина; у части нет эталонного фото — только текстовый канал.
- 14 эталонных фото принадлежат двум slug одновременно — различаются OCR/атрибутами.
- Публичный eval — 3 полевых снимка; полный бенчмарк — на собственной/управленческой выборке.
