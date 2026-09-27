# Выбор технологий (факт реализации)

Цель: instance-level retrieval этикетки/бутылки на CPU-сервере с опциональным GPU через HF Endpoints.  
Ограничение: VPS без x86-64-v2 / без выделенного GPU для backend — тяжёлые модели вынесены в отдельные контейнеры или HF.

## 1. Постановка

Каталог ~2k вин; эталон — студийная бутылка; запрос — полка/стол. Near-duplicates (одна этикетка, разный год/сладость) не решаются одним embedding → **каскад vision + OCR + lexicon + tabular/CE**.

## 2. Итоговый стек

| Слой | Выбрано | Где крутится |
|------|---------|--------------|
| Детекция | Ultralytics YOLO (`label` и `bottle+label`) | `vino_backend`, веса `*.pt` на volume |
| Глобальный embedding A | **SigLIP 2** `google/siglip2-base-patch16-384` (768-d, cosine) | контейнер `siglip2-embed` и/или HF Endpoint; опц. local transformers |
| Глобальный embedding B | **DINOv3** ViT-B/16 | контейнер `dinov3-embed` и/или HF Endpoint |
| ANN | PostgreSQL **pgvector** HNSW (`vector_cosine_ops`) | `vino_postgres` / dev Postgres |
| OCR «лёгкий» | RapidOCR (PP-OCR ONNX) | in-process в backend |
| OCR / VLM | Gemini (proxy), Google Vision, OpenAI, DeepSeek, Qwen-VL (HF), Yandex | по флагам в `pipeline_settings.json` |
| Текстовый matcher | **XGBoost** abs_v14 (`model.json` в git) | in-process |
| Cross-Encoder | отдельный HTTP-сервис | `cross-encoder-matcher` (веса не в git) |
| Отсев | exclusive lexicon (JSON + сорта/винодельни из БД) | in-process |
| Цвет | CIEDE2000 (Lab) ColorDelta; HSV опционально | in-process |
| Backend | Python 3.12, FastAPI, SQLAlchemy, Alembic | `vino_backend` |
| Frontend | React 19 + Vite | `vino_frontend` (nginx static) |
| Оркестрация | Docker Compose `vino_*` + внешний nginx aidispatcher | прод |

## 3. Почему так

- **SigLIP2 + DINOv3, не один CLIP** — VL-семантика этикетки + instance-устойчивость формы; оба уже развёрнуты на сервере конкурса.
- **XGB вместо «только fuzzy»** — обучен на реальных OCR↔label парах (scan id≥800), даёт калиброванный TextScore для FinalScore.
- **Exclusive lexicon** — жёстко режет «белое vs красное» / сорта при шумном OCR до мягкого скоринга.
- **YOLO bottle+label** — склейка полосок и выбор primary по этикетке на корпусе (см. 0.97.x), а не max-area.
- **Отдельные ML-контейнеры** — не раздувать образ backend (CUDA torch не влезает на 35 ГБ диск VPS); prod requirements — CPU torch + RapidOCR.
- **pgvector** — рекомендация организатора; каталог мал; reuse прошлых `search_photo_embeddings` в том же движке.
- **React, не Nuxt** — единый SPA с admin-настройками и историей сканов без SSR-BFF.

## 4. Что сознательно не на горячем пути

| Идея | Статус |
|------|--------|
| SuperPoint + LightGlue на всех кандидатах | опциональная геометрия; не обязательна для match |
| EasyOCR / Surya в prod-образе | отключены (диск/CPU); Rapid + LLM OCR |
| Локальный SigLIP 1.5 GB в git | нет; HF / remote / volume |
| Milvus / Qdrant | избыточно при ~2–4k векторов |
| Классификатор на 2103 класса | каталог растёт; retrieval без переобучения |

## 5. Обучение / дообучение

| Модель | Как |
|--------|-----|
| SigLIP2 fine-tune | `training/siglip2/` → Colab GPU (веса не в git) |
| XGBoost | `training/xgboost/xgboost_train_bundle.zip` + `backend/models/xgboost_text_matcher_abs_v14/` |
| YOLO | локальные прогоны Ultralytics; `*.pt` на volume |
| Cross-Encoder | `research/models/ocr_matcher/train_cross_encoder.py` (вне git-продукта) |

## 6. Конфигурация

- Секреты и URL сервисов — `backend/.env` / `distrib/.env`.
- Пороги и включение движков — `pipeline_settings.json` (UI admin).
- Версия алгоритма — `ALGORITHM_VERSION` (обязательный bump при смене логики).
