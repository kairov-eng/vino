# Vino Svoe — сканер российского вина (РСХБ.Цифра 2026)

Веб-сервис: фото этикетки → карточка из каталога «Своё Вино».

## Быстрый старт (dev)

```bash
# backend :8092
cd backend
cp .env.example .env   # заполните ключи
pip install -r requirements.txt
python run_dev.py

# frontend :8091
cd frontend
npm install
npm run dev
```

Открыть: http://127.0.0.1:8091

## Документация

| Раздел | Путь |
|--------|------|
| Структура монорепо на диске | `../STRUCTURE.md` (локально) |
| ТЗ / архитектура | [`architecture/`](architecture/) |
| Пайплайн findwine | [`docs/findwine-pipeline.md`](docs/findwine-pipeline.md) |
| Конкурс / приёмка | [`docs/contest/`](docs/contest/) |
| **Деплой на vino-svoe.online** | [`distrib/README.md`](distrib/README.md) |
| Обучение SigLIP2 (Colab) / XGBoost | [`training/`](training/) |

## Прод (кратко)

- Контейнеры: `vino_postgres` (pgvector), `vino_backend`, `vino_frontend`
- Media (`/media`, в т.ч. `crop/`) и веса YOLO/XGB: `/var/lib/vino-svoe/...` (volume)
- Дамп каталога без истории сканов: `distrib/sql/vino_catalog.dump.*`
- Обучение: `training/siglip2/` (Colab), `training/xgboost/xgboost_train_bundle.zip`
- В git **нет** SigLIP2/Cross-Encoder весов (`*.safetensors`); XGBoost `model.json` — да

Полная инструкция: **[distrib/README.md](distrib/README.md)**.

## Что не в git

- `.env` и ключи API (есть `.env.example`)
- Фото каталога (`media/`)
- Веса `*.pt` / локальный SigLIP
- Research (`label_detect`, датасеты) — соседняя папка на диске

## Лицензия / конкурс

Учебно-конкурсный проект для ЛЦТ / РСХБ.Цифра 2026.
