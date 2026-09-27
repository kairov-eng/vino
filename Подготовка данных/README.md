# Подготовка данных

Скрипты для сборки данных, на которых живут детекция этикетки (YOLO) и текстовый матчер (XGBoost).
Новичок, который понимает *зачем* crop / YOLO / XGBoost, должен суметь всё запустить по этой инструкции.

Обучение SigLIP2 (Colab) и готовый train-bundle XGBoost — в соседней папке [`../training/`](../training/).

## Что внутри

| Папка | Зачем |
|-------|--------|
| [`crop/`](crop/) | Нарезать этикетки с фото каталога / iPhone через Gemini, записать `crop/` + `cropjson/` и поля в БД |
| [`yolo/`](yolo/) | Из `cropjson/` собрать YOLO-датасет (классы bottle/label) и синтетику |
| [`xgboost/`](xgboost/) | Из истории сканов (`search_photos`) собрать пары OCR↔вино и обучить XGBoost |

Поток данных (упрощённо):

```
фото каталога / HEIC
        │
        ▼
   crop/  (Gemini)  ──►  media/.../crop + cropjson  (+ wines.crop в Postgres)
        │
        ▼
   yolo/  ──►  images/labels + data.yaml  ──►  обучение YOLO (Colab-скрипт)

история findwine (search_photos.status)
        │
        ▼
   xgboost/  ──►  parquet/jsonl  ──►  model.json (текстовый матчер)
```

## Общие требования

1. Python 3.11+ (тот же, что для `backend`).
2. Работающая Postgres с каталогом (`DATABASE_URL` в `backend/.env`) — нужна для **crop каталога** и **XGBoost**.
3. Ключи Gemini в `backend/.env` (или локальный `.env` рядом со скриптами) — нужны для **crop**.
4. Фото каталога на диске: по умолчанию `<корень_монорепо>/media/uploads`
   (рядом с папкой `vino-svoe`, не внутри неё).

Скрипты сами ищут `.env` в таком порядке:

1. `crop/.env` или `Подготовка данных/.env`
2. иначе `vino-svoe/backend/.env`

### Быстрая установка зависимостей

Из корня `vino-svoe` (рекомендуется venv backend):

```powershell
cd C:\dev\Vino2026\vino-svoe
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "Подготовка данных/requirements.txt"
# для crop каталога также нужен доступ к БД — обычно уже есть через backend:
pip install -r backend/requirements.txt
```

Скопируйте пример переменных:

```powershell
copy "Подготовка данных\.env.example" "Подготовка данных\.env"
# или допишите те же ключи в backend\.env
```

Минимум в `.env`:

```env
DATABASE_URL=postgresql+psycopg2://USER:PASS@127.0.0.1:5432/vino
GEMINI_API_KEY_1=...
GEMINI_MODEL=gemini-3.6-flash
UPLOADS_DIR=C:\dev\Vino2026\media\uploads
```

---

## 1. Crop каталога (`crop/`)

### Зачем

Без кропов нет эталонных этикеток в `/media/crop/` и нет разметки `cropjson/` для YOLO.
Скрипт `2.crop_wines.py` берёт строки из таблицы `wines`, находит файл в `uploads/`, зовёт Gemini и пишет:

- `uploads/crop/<файл>_crop.png`
- `uploads/cropjson/<файл>.json`
- статусы `wines.crop`, текст `wines.label`, структура `wines.label_ocr`

### Перед запуском

- Postgres запущена, миграции применены (`alembic upgrade head`).
- В `uploads/` лежат фото каталога (имена как в `wines.photo_name` / sitemap).
- В `.env` есть `GEMINI_API_KEY_1` (или Vertex — см. комментарии в `crop/gemini_client.py`).

### Запуск

```powershell
cd "C:\dev\Vino2026\vino-svoe\Подготовка данных\crop"

# тест на одной карточке
python 2.crop_wines.py --id 3099

# батч
python 2.crop_wines.py --limit 100

# крутить, пока есть необработанные
python 2.crop_wines.py --infinitive 1

# другой каталог фото
python 2.crop_wines.py --uploads C:\dev\Vino2026\media\uploads --limit 50
```

Логи: `crop/logs/crop_wines_*.log`.

### HEIC с телефона (без БД)

```powershell
python "3.crop_heic_folder.py" C:\photos\iphone --limit 20
python "3.crop_heic_folder.py" C:\photos\iphone --file IMG_1234.HEIC
```

Пишет `crop/` и `cropjson/` **внутри указанной папки** (не в media каталога).

### Локальный тест Gemini на папке картинок

```powershell
python detect_bottle_label.py C:\dev\Vino2026\temp
```

Результат с рамками — в `crop/results/`.

Подробные комментарии — в шапках самих `.py` файлов.

---

## 2. YOLO (`yolo/`)

### Зачем

YOLO в проде режет бутылку/этикетку на фото пользователя.
Датасет собирается из полевых фото + `cropjson/` (разметка Gemini).

### Цепочка

1. Разметить полевые фото: `crop/3.crop_heic_folder.py` → `<папка>/cropjson/`.
2. Собрать реальный сет: `8.prepare_yolo_dataset.py`.
3. (Опционально) синтетика: `5.generate_synthetic_yolo_bottle_label.py` или `6.generate_synthetic_yolo.py`.
4. Обучение: `15.colab_train_yolo_bottle_label.py` (удобнее в Google Colab).

Пути по умолчанию (от корня монорепо `Vino2026/`):

| Аргумент | По умолчанию |
|----------|----------------|
| полевые фото / cropjson | `uploads.my/` |
| каталог эталонов | `media/uploads/` |
| синтетика | `uploads.syntetic/yolo/` |

Создайте `uploads.my/` рядом с `vino-svoe`, положите туда HEIC/JPG и получите `cropjson/` скриптом 3.

### Запуск

```powershell
cd "C:\dev\Vino2026\vino-svoe\Подготовка данных\yolo"

# только labels по уже существующему split images/
python "8.prepare_yolo_dataset.py" --labels-only

# полный экспорт: HEIC→JPG, train/val, labels, data.yaml
python "8.prepare_yolo_dataset.py" --full --val-ratio 0.2 --seed 42 --uploads C:\dev\Vino2026\uploads.my

# синтетика (нужны cropjson каталога + полевые images)
python "5.generate_synthetic_yolo_bottle_label.py" --n 500 --seed 42
```

Выход реального сета: `<uploads>/yolo/{images,labels}/{train,val}` + `data.yaml`.

Обучение в Colab — откройте `15.colab_train_yolo_bottle_label.py` и следуйте комментариям в файле (загрузка zip датасета, ultralytics).

---

## 3. XGBoost (`xgboost/`)

### Зачем

После SigLIP2 (Top‑N кандидатов) модель оценивает `P(то же вино)` по тексту OCR запроса и полям карточки.
Обучается на парах из реальных сканов в `search_photos`.

### Перед запуском

- В БД есть история сканов с заполненным `status` (JSON пайплайна findwine).
- `backend/.env` с рабочим `DATABASE_URL`.
- Пакеты: `pip install -r xgboost/requirements.txt` (или общий `requirements.txt` этой папки).

### Основной пайплайн (рекомендуется)

```powershell
cd "C:\dev\Vino2026\vino-svoe\Подготовка данных\xgboost"

# 1) БД → data/raw + data/processed + data/features
python prepare_dataset.py

# 2) обучение → xgboost_text_matcher/model.json
python train_xgboost.py
```

Артефакты пишутся **в эту же папку** `xgboost/`:

```
xgboost/
  data/
    raw/
    processed/
    features/
  xgboost_text_matcher/
    model.json
    feature_names.json
    metrics.json
    …
```

Готовую модель для продакшена обычно копируют в `backend` / volume `models` (см. `distrib/README.md`).

Полезные флаги:

```powershell
python prepare_dataset.py --limit 200          # отладка
python prepare_dataset.py --all-ocr-engines    # аугментация по OCR-движкам
python train_xgboost.py --max-depth 4 --n-estimators 1500
```

### Альтернативные сборщики датасета

В папке также лежат более новые/экспериментальные скрипты — если нужен другой протокол пар:

| Скрипт | Когда |
|--------|--------|
| `prepare_xgboost_dataset_v2.py` | v2-признаки / другие каналы OCR |
| `prepare_xgboost_dataset_query_ocr.py` | акцент на query-OCR |
| `prepare_xgboost_synthetic.py` | синтетические пары |
| `add_hard_reject_features.py` | доп. hard-reject признаки |

Смотрите `--help` у каждого. Базовый путь «новичка»: `prepare_dataset.py` → `train_xgboost.py`.

Больше деталей: [`xgboost/README.md`](xgboost/README.md) (пути в нём исторические — рабочие команды выше актуальнее).

---

## Частые проблемы

| Симптом | Что проверить |
|---------|----------------|
| `DATABASE_URL` не найден | Есть ли `backend/.env`, активирован ли venv |
| Gemini 401 / quota | `GEMINI_API_KEY_*`, лимит RPM (скрипт сам троттлит ≤15/мин на ключ) |
| crop=0, файл не найден | Имя фото в `uploads/` vs `wines.photo_name` / sitemap |
| YOLO пустые labels | В `cropjson` должен быть `status == 1` и боксы `*_box_2d` |
| XGBoost «нет сканов» | В `search_photos` мало строк со `status` / trusted sources |

## Что не коммитить

- `.env`, логи (`**/logs/`), `results/`, сырые датасеты `data/`, большие `uploads*`
- Веса YOLO `*.pt` (кроме того, что уже лежит в `distrib` / server volume)

В git — только скрипты, промпты и эта инструкция.
