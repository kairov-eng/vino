# План и статус реализации

Исходный план (3 этапа) выполнен в виде единого продукта **React + FastAPI + pgvector**, а не Nuxt/`services/ml`. Ниже — соответствие целям этапов и оставшиеся работы.

## Этап 1. Данные и индекс — сделано

| Цель | Статус |
|------|--------|
| Каталог ~2103 вин в PostgreSQL | да (`wines`, `wineries`, sitemap) |
| Сопоставление фото slug↔uploads | да (`media.py` / crop_wines) |
| Кропы этикеток | да (`media/crop`, `wines.crop`) |
| Embeddings SigLIP2 / DINOv3 | да (pgvector HNSW) |
| Дамп для сервера без истории сканов | да (`distrib/sql/`) |

Инструменты офлайн-подготовки — в соседнем `research/label_detect/` (не обязательно в git продукта).

## Этап 2. Поиск и диагностика — сделано

| Цель | Статус |
|------|--------|
| End-to-end `POST /api/findwine` | да, алгоритм 0.97.x |
| YOLO label / bottle+label | да |
| OCR cascade + exclusive + XGB + CE | да |
| `/v1/eval/predict` + `participant_test.sh` | да |
| Диагностика | admin settings + `status` в ответе + история сканов |
| UI сканера | React HomePage |

## Этап 3. Продукт и деплой — сделано

| Цель | Статус |
|------|--------|
| Mobile-first UI | да |
| Аналоги после поиска | да |
| Site gate | да |
| Docker `vino_backend` / `vino_frontend` / `vino_postgres` | да |
| Прод https://vino-svoe.online | да |
| Документация architecture + distrib + findwine-pipeline | да (этот апдейт) |
| Обучение SigLIP/XGB в репо | да (`training/`) |

## Бэклог (не блокирует приёмку)

1. Поднять Langfuse на сервере заново (образы снимались ради диска).
2. Больше полевых фото / отчёт `reports/public_eval_*.md` с свежими цифрами.
3. Опционально: «сомелье»-диалог (сейчас retention = аналоги + история).
4. Тонкая калибровка порогов под приватный набор организаторов.
5. Ускорение: меньше внешних OCR на eval-пути; CDN для media при росте.

## Критерии «готово к защите»

См. чеклист [`../docs/contest/ACCEPTANCE.md`](../docs/contest/ACCEPTANCE.md):

- [x] UI на домене
- [x] health / findwine / media / eval endpoint
- [x] каталог + embeddings
- [x] Docker + README/ARCHITECTURE
- [ ] видео ≤ 7 мин и презентация по шаблону (контент команды)
- [ ] финальный прогон метрик на контрольной выборке
