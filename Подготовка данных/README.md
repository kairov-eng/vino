# Подготовка данных

Скрипты для сборки данных под детекцию этикетки (**YOLO**) и текстовый матчер (**XGBoost**).
Новичок, который понимает *зачем* crop / YOLO / XGBoost, должен суметь всё запустить по этой инструкции.

Обучение SigLIP2 (Colab) и готовый train-bundle XGBoost — в [`../training/`](../training/).

## Быстрые ссылки

| Задача | Куда смотреть |
|--------|----------------|
| **YOLO: подготовка данных → Colab** | [§ YOLO](#2-yolo--подготовка-данных-и-обучение-в-colab) |
| **XGBoost: подготовка данных → обучение** | [§ XGBoost](#3-xgboost--подготовка-данных-и-обучение) |
| Crop каталога / HEIC (разметка Gemini) | [§ Crop](#1-crop-каталога-crop) |

## Что внутри

| Папка | Зачем |
|-------|--------|
| [`crop/`](crop/) | Gemini: crop этикеток каталога / iPhone → `crop/` + `cropjson/` (+ поля в БД) |
| [`yolo/`](yolo/) | Сборка YOLO-датасета (bottle/label), синтетика, скрипт обучения для Colab |
| [`xgboost/`](xgboost/) | Пары OCR↔вино из `search_photos` → признаки → `model.json` |

```
полевые фото / HEIC
        │
        ▼
 crop/3.crop_heic_folder.py  ──►  cropjson/  (боксы bottle + label)
        │
        ▼
 yolo/8.prepare_yolo_dataset.py  ──►  yolo/{images,labels}/ + data.yaml
        │                              (+ опционально синтетика 5 / 6)
        ▼
 zip → Google Drive → yolo/15.colab_train_yolo_bottle_label.py  ──►  best.pt

история findwine (search_photos.status)
        │
        ▼
 xgboost/prepare_dataset.py  ──►  data/features/*.parquet
        │
        ▼
 xgboost/train_xgboost.py  ──►  xgboost_text_matcher/model.json
```

## Общие требования

1. Python 3.11+ (тот же, что для `backend`).
2. Postgres + `DATABASE_URL` в `backend/.env` — для **crop каталога** и **XGBoost**.
3. Ключи Gemini — для **crop** / разметки полевых фото.
4. Фото каталога: по умолчанию `<корень_монорепо>/media/uploads` (рядом с `vino-svoe`).

`.env` ищется в порядке: `crop/.env` → `Подготовка данных/.env` → `vino-svoe/backend/.env`.

```powershell
cd C:\dev\Vino2026\vino-svoe
pip install -r "Подготовка данных/requirements.txt"
copy "Подготовка данных\.env.example" "Подготовка данных\.env"
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

Нужен для эталонных кропов в проде (`/media/crop/`) и для разметки `cropjson/` (в т.ч. как вход синтетики YOLO).

| Скрипт | Роль |
|--------|------|
| `2.crop_wines.py` | Каталог из Postgres (`wines`) → Gemini → `crop/` + `cropjson/` + `wines.crop` |
| `3.crop_heic_folder.py` | Папка HEIC/JPG **без БД** → `cropjson/` (это вход для YOLO с полевых фото) |
| `detect_bottle_label.py` | Локальный тест Gemini на папке; также библиотека для 2/3 |
| `gemini_client.py` | AI Studio / Vertex |

```powershell
cd "C:\dev\Vino2026\vino-svoe\Подготовка данных\crop"

python 2.crop_wines.py --id 3099
python 2.crop_wines.py --limit 100
python 2.crop_wines.py --infinitive 1

# полевые фото для YOLO (без БД)
python "3.crop_heic_folder.py" C:\dev\Vino2026\uploads.my --infinitive 1
```

Подробности — в шапках `.py`.

---

## 2. YOLO — подготовка данных и обучение в Colab

Прод-модель детектирует **два класса**: `0 = bottle`, `1 = label`.  
Ниже — полный путь от сырых фото до `best.pt` в Google Colab.

### 2.1. Какие скрипты за что отвечают

| # | Скрипт | Когда запускать | Что на выходе |
|---|--------|-----------------|---------------|
| A | [`crop/3.crop_heic_folder.py`](crop/3.crop_heic_folder.py) | Сначала: разметить полевые фото | `<папка>/cropjson/*.json` с `bottle_box_2d`, `label_box_2d`, `status` |
| B | [`yolo/8.prepare_yolo_dataset.py`](yolo/8.prepare_yolo_dataset.py) | Собрать **реальный** датасет | `<uploads>/yolo/images|labels/{train,val}` + `data.yaml` |
| C | [`yolo/5.generate_synthetic_yolo_bottle_label.py`](yolo/5.generate_synthetic_yolo_bottle_label.py) | Опционально: синтетика **bottle+label** (как в проде) | `uploads.syntetic/yolo/…` |
| D | [`yolo/6.generate_synthetic_yolo.py`](yolo/6.generate_synthetic_yolo.py) | Опционально: синтетика **только label** (1 класс) | тот же каталог, другой `data.yaml` |
| E | [`yolo/15.colab_train_yolo_bottle_label.py`](yolo/15.colab_train_yolo_bottle_label.py) | В **Google Colab** после загрузки zip на Drive | `best.pt` / `last.pt` |

Для прод-YOLO (bottle+label) используйте цепочку **A → B** (и при необходимости **C**), затем **E**.  
Скрипт **6** — если учите модель только на этикетку (не текущий прод-контракт).

### 2.2. Подготовка реального датасета (шаг за шагом)

**Шаг 1.** Создайте папку полевых фото (рядом с `vino-svoe`):

```text
C:\dev\Vino2026\uploads.my\
  IMG_….HEIC / .jpg / …
```

**Шаг 2.** Разметьте Gemini (боксы бутылки и этикетки):

```powershell
cd "C:\dev\Vino2026\vino-svoe\Подготовка данных\crop"
python "3.crop_heic_folder.py" C:\dev\Vino2026\uploads.my --infinitive 1
```

Появятся `uploads.my/cropjson/*.json`. В обучение попадают записи со `status == 1` и валидными боксами.

**Шаг 3.** Соберите YOLO-сет:

```powershell
cd "C:\dev\Vino2026\vino-svoe\Подготовка данных\yolo"

# первый раз / пересборка images + labels + split
python "8.prepare_yolo_dataset.py" --full --val-ratio 0.2 --seed 42 --uploads C:\dev\Vino2026\uploads.my

# если JPG уже лежат в yolo/images/{train,val} — только переписать labels
python "8.prepare_yolo_dataset.py" --labels-only --uploads C:\dev\Vino2026\uploads.my
```

Структура результата:

```text
uploads.my/yolo/
  images/train|val/*.jpg
  labels/train|val/*.txt   # строки: "cls xc yc w h"  (cls 0=bottle, 1=label)
  data.yaml
```

### 2.3. Синтетика (опционально)

Нужны:

- каталог эталонов с `cropjson/` — обычно `media/uploads/` (после `2.crop_wines.py`);
- фоны — `uploads.my/yolo/images/{train,val}` (после шага 3).

```powershell
cd "C:\dev\Vino2026\vino-svoe\Подготовка данных\yolo"

# 2 класса (как в проде) — рекомендуется
python "5.generate_synthetic_yolo_bottle_label.py" --n 1500 --seed 42

# только этикетка (1 класс)
python "6.generate_synthetic_yolo.py" --n 1500 --seed 42

# маленький прогон + preview
python "5.generate_synthetic_yolo_bottle_label.py" --n 50 --preview 8
```

Выход по умолчанию: `C:\dev\Vino2026\uploads.syntetic\yolo\`.

Можно обучать только на реальном сете, только на синтетике или объединить папки вручную перед упаковкой в zip (внутри должна остаться структура `yolo/images|labels/...` + `data.yaml`).

### 2.4. Упаковка для Colab

Упакуйте каталог **`yolo/`** (тот, где `images/`, `labels/`, `data.yaml`) в zip так, чтобы внутри архива был префикс `yolo/`:

```powershell
# пример: реальный сет
Compress-Archive -Path C:\dev\Vino2026\uploads.my\yolo -DestinationPath C:\dev\Vino2026\yolo_bottle_label.zip -Force
```

Загрузите zip на Google Drive, например:

```text
MyDrive/Vino/yolo_bottle_label.zip
```

(путь по умолчанию в скрипте Colab — именно такой; при другом пути поправьте `ZIP_PATH` в ячейке).

### 2.5. Обучение в Google Colab

1. Откройте [Google Colab](https://colab.research.google.com/), Runtime → **GPU**.
2. Создайте ноутбук и **скопируйте содержимое** [`yolo/15.colab_train_yolo_bottle_label.py`](yolo/15.colab_train_yolo_bottle_label.py) в ячейки (по блокам из комментариев файла) **или** загрузите файл и выполняйте секции подряд.
3. Скрипт делает:
   - `nvidia-smi`, `pip install ultralytics`
   - монтирует Drive
   - распаковывает zip → `/content/datasets/yolo_bottle_label/yolo/`
   - переписывает `data.yaml` под пути Colab (`names: 0 bottle, 1 label`)
   - обучает `YOLO("yolo11n.pt")` → веса на Drive:  
     `MyDrive/Vino/runs/bottle_label_<timestamp>/weights/best.pt`
4. Параметры по умолчанию: `epochs=80`, `imgsz=640`, `batch=16` (при OOM уменьшите `batch` до 8/4; для мелких объектов на полке можно `imgsz=960`).
5. Если обучение оборвалось — в отдельной ячейке `model.train(resume=True)` от `last.pt` (см. комментарий в конце файла 15).

После обучения скопируйте `best.pt` в прод:

- локально: `backend/models/yolo_bottle_label/best.pt` (или `yolo/`, см. `distrib/README.md`);
- на сервер:  
  `rsync … aidispatcher:/var/lib/vino-svoe/models/yolo_bottle_label/`

---

## 3. XGBoost — подготовка данных и обучение

XGBoost в пайплайне — `P(same wine)` для пары **OCR запроса ↔ карточка вина** после SigLIP2 top‑N.  
Обучается на реальной истории сканов (`search_photos`).

### 3.1. Какие скрипты за что отвечают

| Скрипт | Роль | Нужен новичку? |
|--------|------|----------------|
| [`prepare_dataset.py`](xgboost/prepare_dataset.py) | БД → `data/raw` + `processed` + **`features/*.parquet`** | **Да — основной сборщик** |
| [`train_xgboost.py`](xgboost/train_xgboost.py) | Обучение на parquet → `xgboost_text_matcher/model.json` | **Да** |
| [`text_features.py`](xgboost/text_features.py) | Признаки OCR↔label (общий модуль с продом) | библиотека |
| [`eval_utils.py`](xgboost/eval_utils.py) | Метрики | библиотека |
| [`prepare_xgboost_dataset_v2.py`](xgboost/prepare_xgboost_dataset_v2.py) | Альтернативный протокол пар / каналы OCR | эксперименты |
| [`prepare_xgboost_dataset_query_ocr.py`](xgboost/prepare_xgboost_dataset_query_ocr.py) | Акцент на query-OCR | эксперименты |
| [`prepare_xgboost_synthetic.py`](xgboost/prepare_xgboost_synthetic.py) | Синтетические пары | эксперименты |
| [`add_hard_reject_features.py`](xgboost/add_hard_reject_features.py) | Доп. hard-reject признаки | эксперименты |
| [`fp_synthetic.py`](xgboost/fp_synthetic.py) / [`hard_reject_features.py`](xgboost/hard_reject_features.py) | Вспомогательные модули | библиотека |

Базовый путь: **`prepare_dataset.py` → `train_xgboost.py`**.

Готовый архив со скриптами + parquet для воспроизведения обучения (без своей БД) также лежит в [`../training/xgboost/`](../training/xgboost/).

### 3.2. Что делает `prepare_dataset.py`

1. Читает `search_photos` (+ эталоны вин) через `DATABASE_URL` из `backend/.env`.
2. Берёт OCR запроса (сырые строки, без LLM): приоритет `google_vision` → `gemini` → `openai` (настраивается).
3. Кандидаты — SigLIP2 top‑N из `status.steps.candidates.siglip2`.
4. Разметка:
   - trusted `matched_wine.source` (`final_score`, `final_score2` по умолчанию) → positive / negatives;
   - `eval.false_positive=1` → hard negative;
   - FN без политики → скан пропускается.
5. Split по группам (id найденного вина), без утечки между train/val/test.
6. Пишет:

```text
Подготовка данных/xgboost/data/
  raw/scans.jsonl, wines.jsonl
  processed/{train,validation,test}.jsonl   # пары text1/text2 (также для CE)
  features/{train,validation,test}.parquet  # числовые признаки для XGB
  dataset_stats.json
```

### 3.3. Запуск

```powershell
cd "C:\dev\Vino2026\vino-svoe\Подготовка данных\xgboost"
pip install -r requirements.txt   # или общий ../requirements.txt

# 1) данные
python prepare_dataset.py
python prepare_dataset.py --limit 200                    # отладка
python prepare_dataset.py --all-ocr-engines              # пары по каждому OCR
python prepare_dataset.py --sources final_score,final_score2,siglip2

# 2) обучение
python train_xgboost.py
python train_xgboost.py --max-depth 4 --n-estimators 1500
python train_xgboost.py --drop-features rel_cos_rank,siglip_cosine   # абляция
```

Артефакты обучения:

```text
xgboost/xgboost_text_matcher/
  model.json
  feature_names.json
  metrics.json
  feature_importance.json
  train_config.json
  predictions_test.csv
```

### 3.4. Куда класть модель в прод

Прод читает каталог из `XGB_MODEL_DIR` (compose: `/app/models/xgboost_text_matcher_abs_v14`).

```powershell
# пример: обновить локальную прод-копию
Copy-Item -Recurse -Force `
  ".\xgboost_text_matcher\*" `
  "..\..\backend\models\xgboost_text_matcher_abs_v14\"

# на сервер (см. distrib/README.md)
# rsync -avP ./backend/models/xgboost_text_matcher_abs_v14/ `
#   aidispatcher:/var/lib/vino-svoe/models/xgboost_text_matcher_abs_v14/
```

Нужен пакет `scikit-learn` в образе backend (`XGBClassifier`), иначе `xgb_match` падает с ошибкой про sklearn.

Доп. детали (исторические пути): [`xgboost/README.md`](xgboost/README.md).

---

## Частые проблемы

| Симптом | Что проверить |
|---------|----------------|
| `DATABASE_URL` не найден | `backend/.env`, venv |
| Gemini 401 / quota | `GEMINI_API_KEY_*`, RPM ≤15/мин на ключ |
| crop=0 / файл не найден | имя в `uploads/` vs `wines.photo_name` |
| YOLO пустые labels | в `cropjson`: `status == 1` и боксы `*_box_2d` |
| Colab: нет zip | путь `ZIP_PATH` = фактический файл на Drive |
| Colab: OOM | `batch=8` или `4` |
| XGBoost «нет сканов» | мало `search_photos` с trusted `matched_wine` |
| Прод: XGB `ok:false` + sklearn | установить `scikit-learn` в `requirements.prod.txt` / образ |

## Что не коммитить

- `.env`, `**/logs/`, `results/`, сырые `data/`, `uploads*`, веса `*.pt`

В git — скрипты, промпты и эта инструкция.
