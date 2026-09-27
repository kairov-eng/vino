> **Актуальный запуск:** см. [`../README.md`](../README.md) (раздел XGBoost).
> Скрипты лежат в `vino-svoe/Подготовка данных/xgboost/`.
> Данные и `model.json` по умолчанию пишутся **сюда же** (`./data`, `./xgboost_text_matcher`).
> `DATABASE_URL` берётся из `vino-svoe/backend/.env`.

# OCR Text Matcher — XGBoost + Cross-Encoder

Второй этап поиска вина: после SigLIP2 retrieval (Top-20) модель оценивает
`P(same wine)` для пары **(OCR текст этикетки запроса, эталон вина из каталога)**.
Реализованы оба подхода из ТЗ (`temp/ТЗ на подготовку данных для обучения XGBoost и трансформера.txt`):

| Подход | Вход | Выход | Где |
|---|---|---|---|
| **XGBoost** (feature-based) | 94 числовых признака пары + SigLIP cosine | вероятность 0..1 | `models/xgboost_text_matcher/` |
| **Cross-Encoder** (Transformer) | `text1` = строки OCR, `text2` = эталон вина | вероятность 0..1 | `models/cross_encoder_text_matcher/` |

Оба обучаются на **одном и том же** наборе пар (`models/data/processed/*.jsonl`) — ТЗ п.26.

## Структура

```
C:\dev\Vino2026\models\
├── ocr_matcher\                      # скрипты
│   ├── prepare_dataset.py            # 1) БД → пары → split → признаки
│   ├── train_xgboost.py              # 2) обучение XGBoost
│   ├── train_cross_encoder.py        # 3) fine-tuning Cross-Encoder
│   ├── text_features.py              # нормализация + признаки (общий модуль, нужен и в production)
│   ├── eval_utils.py                 # метрики: ROC-AUC, F1, Top-k, latency
│   ├── requirements.txt
│   └── README.md
├── data\                             # данные для обучения
│   ├── raw\scans.jsonl               # 1 строка = скан: OCR, кандидаты SigLIP2, найденное вино, FP/FN
│   ├── raw\wines.jsonl               # эталоны вин (label, name, winery, category, grape, region)
│   ├── processed\{train,validation,test}.jsonl   # пары для Cross-Encoder (text1/text2/label + метаданные)
│   ├── features\{train,validation,test}.parquet  # признаки для XGBoost
│   └── dataset_stats.json
├── xgboost_text_matcher\             # обученная модель XGBoost
│   ├── model.json, feature_names.json, metrics.json, feature_importance.json,
│   └── train_config.json, predictions_test.csv
└── cross_encoder_text_matcher\       # обученный Cross-Encoder (HF-формат)
    ├── config.json, model.safetensors, tokenizer.*, 
    └── metrics.json, train_config.json, predictions_test.csv
```

## Как запускать (3 скрипта)

Окружение — то же, что у backend (`python`, доступ к PostgreSQL через `vino-svoe/backend/.env` → `DATABASE_URL`).
Дополнительно нужен `xgboost` (`pip install xgboost`), остальное (`sentence-transformers`, `torch`, `datasets`, `rapidfuzz`) уже установлено.

```powershell
Set-Location C:\dev\Vino2026\models\ocr_matcher

# 1. Подготовка данных (~15 с): search_photos → models/data/*
python prepare_dataset.py

# 2. XGBoost (~10 с, CPU) → models/xgboost_text_matcher/
python train_xgboost.py

# 3. Cross-Encoder (CPU ~45–60 мин; GPU — минуты) → models/cross_encoder_text_matcher/
python train_cross_encoder.py
```

Полезные варианты:

```powershell
# данные: добавить cosine-fallback (`matched_wine.source=siglip2`) как positives — шумнее, но в 1.8 раза больше сканов
python prepare_dataset.py --sources final_score,final_score2,siglip2
# данные: пары для каждого доступного OCR-движка скана (Google Vision + Gemini + OpenAI) — аугментация
python prepare_dataset.py --all-ocr-engines
# данные: FN-сканы считать верно найденными (показанное вино = positive)
python prepare_dataset.py --fn-policy positive

# XGBoost: гиперпараметры / абляция признаков
python train_xgboost.py --max-depth 4 --learning-rate 0.02 --n-estimators 1500
python train_xgboost.py --drop-features rel_cos_rank,rel_cos_delta_top,rel_cos_z,siglip_cosine   # «только текст»

# Cross-Encoder: все negatives, больше эпох (лучше на GPU)
python train_cross_encoder.py --neg-per-pos 0 --epochs 4 --max-length 256
# Cross-Encoder: маленькая модель (в ~2 раза быстрее)
python train_cross_encoder.py --model nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large
# только оценить уже сохранённую модель (+ сравнение с XGBoost)
python train_cross_encoder.py --eval-only
```

## Источник данных и разметка

Таблица `search_photos` (история поиска), поле `status` (JSONB):

- **OCR запроса** — только сырые строки, без LLM-категоризации (`json.*` не используется):
  `steps.ocr.variants.google_vision.lines[].text` → если нет, `gemini.lines` → `openai.lines`
  (приоритет `--ocr-engines`). Строки склеиваются через `\n`; для Cross-Encoder — через ` | `.
- **Кандидаты** — только SigLIP2: `steps.candidates.siglip2.items[]` (`id`, `cosine_similarity`), Top-20.
  `candidates_cosine` **не** используется как основной источник, т.к. там max(siglip2, dinov3).
- **Эталон вина** — `wines.label` (текст этикетки каталога построчно) + `name`, `winery`, `category`, `grape_variety`, `region`.
- **Метки:**

| Ситуация в записи | Разметка |
|---|---|
| `eval.false_positive=1` | найденное вино → **0 (hard negative)**; остальные кандидаты не размечаются (истинное вино неизвестно) |
| `eval.false_negative=1` | пропуск (истинное вино неизвестно); `--fn-policy positive` — считать показанное вино верным |
| флагов нет, `matched_wine.source ∈ {final_score, final_score2}` | найденное вино → **1**, остальные 19 SigLIP-кандидатов → **0** |
| `source=siglip2` (cosine top-1 без текстового подтверждения, в UI «не найдено») | пропуск (см. `--sources`) |
| нет OCR / нет кандидатов / нет `matched_wine` | пропуск |

- **Дедупликация**: одно фото (sha256) могло сканироваться несколько раз — остаётся размеченный (FP/FN) или самый новый скан.
- **Split** 70/15/15 **по группам** (`group_id` = id найденного вина), а не по строкам — чтобы одно и то же вино
  не оказалось в train и test (ТЗ п.7). Каталог общий: вино может быть negative в train и positive в test — это
  соответствует production (каталог фиксирован, новые фото).

Текущий срез (2026‑09‑22): 786 сканов в БД → 347 принято (276 `siglip2` без подтверждения, 97 без OCR, 19 без совпадения, 5 FN, 42 дубликата) → **6 522 пары**, 325 positive, 22 hard negative (FP), 6 175 negative. Train 243 сканов / val 52 / test 52.

> Важно: positives размечены **текущим пайплайном** (FinalScore) плюс ручные FP/FN. Модели учатся воспроизводить его решения и исправления. Чем больше ручной разметки FP/FN в истории, тем полезнее датасет — стоит размечать регулярно и пересобирать данные.

## Признаки XGBoost (`text_features.py`)

94 признака, `FEATURE_NAMES` фиксирует порядок (сохраняется в `feature_names.json`):

- **строковые**: Levenshtein, Jaro-Winkler, `fuzz.ratio/partial/token_sort/token_set/WRatio`, char n-gram Jaccard (2/3/4), длины и их отношения;
- **токены**: Jaccard, overlap, fuzzy-покрытие токенов эталона OCR‑ом и наоборот (mean/max/доля ≥0.8), отдельно для «длинных» токенов (названия);
- **числа**: год (есть/совпал/не совпал/мин. разница), прочие числа, крепость `%`, объём `л/ml`;
- **посимвольные**: совпадение только цифр / только букв, доля кириллицы;
- **структурные поля вина**: покрытие `name`, `winery`, `grape`, `region`, лучшая строка OCR для имени/винодельни;
- **лексика**: цвет (красное/белое/розовое/… vs `category`), сладость (сухое/полусладкое/брют…), игристое — match/mismatch;
- **embedding**: `siglip_cosine`;
- **относительные по группе кандидатов** (`rel_*`): ранг/отрыв от лидера/z‑score по cosine, по `token_set_ratio`, по покрытию токенов, по имени.

Из-за относительных признаков в production признаки считаются **сразу для всех Top‑N кандидатов** одного запроса:

```python
from text_features import WineRef, group_features, FEATURE_NAMES
rows = group_features(ocr_text, [WineRef(...), ...], [cos1, cos2, ...])
X = pd.DataFrame(rows)[feature_names]          # feature_names.json из папки модели
probs = XGBClassifier().load_model(...).predict_proba(X)[:, 1]
```

Cross-Encoder в production:

```python
from sentence_transformers import CrossEncoder
ce = CrossEncoder(r"C:\dev\Vino2026\models\cross_encoder_text_matcher")   # activation=Sigmoid сохранена в config
probs = ce.predict([[text1, wine.cross_encoder_text()] for wine in top_k], batch_size=32)
```

## Результаты

См. `metrics.json` в папках моделей (обновляются при каждом запуске). Сводка — в конце вывода `train_cross_encoder.py`
(таблица XGBoost / Cross-Encoder / XGBoost Top‑3 → CE на одинаковых test-парах) и в разделе ниже.

### Сводка test (49 сканов с positive)

| Модель | Top-1 | Top-3 | Top-5 | MRR | ROC-AUC | Latency (CPU) |
|---|---|---|---|---|---|---|
| SigLIP cosine (baseline) | 69.4% | 79.6% | — | — | — | — |
| **XGBoost** | **73.5%** | **85.7%** | **93.9%** | **0.81** | **0.95** | ~7 ms / 20 cand |
| Cross-Encoder | 59.2% | 73.5% | 83.7% | 0.69 | 0.81 | ~1.2 s / 20 cand |
| mean(XGB, CE) | 75.5% | 81.6% | 89.8% | 0.82 | 0.93 | — |
| XGB Top-3 → CE | 69.4% | 85.7% | 93.9% | 0.79 | — | — |

На текущих ~350 сканах **XGBoost заметно сильнее CE** и почти бесплатен по latency.
Cross-Encoder (85 мин на CPU, 3 эпохи, MiniLM-L12) улучшил zero-shot (Top-1 47%→59%), но пока не обгоняет cosine baseline и XGBoost.
Для production на этом срезе данных разумен путь: **SigLIP → XGBoost**. CE имеет смысл дообучать позже на GPU / с большим датасетом (больше FP/FN-разметки).

## Выбор пути (анализ ТЗ)

Из предложений ChatGPT взято:

- **Сначала XGBoost** на признаках пары + SigLIP cosine — почти бесплатно по latency (CPU, единицы мс), хорошо
  подходит для fusion текстовых сигналов с cosine. Веса вместо ручной формулы `0.45·text + 0.55·cos` подбирает модель.
- **Cross-Encoder** как второй этап только над Top‑k (batch inference), а не над всем каталогом. Маленькая
  multilingual MiniLM, не LLM.
- **Один датасет для обоих** подходов, split по вину, hard negatives — **реальные** (все Top‑20 SigLIP‑кандидаты
  визуально похожи, плюс ручные FP), **синтетическая порча OCR не делается** (реальных OCR‑ошибок достаточно, ТЗ рекомендует отложить).

Отличия от ТЗ и почему:

- Не генерируем «easy negatives» из случайных вин: в production XGBoost/CE видят только SigLIP‑кандидатов, поэтому
  обучаемся на том же распределении (ТЗ само отмечает, что negatives лучше брать из Top‑N).
- Добавлены **относительные признаки по группе** (`rel_*`) — задача по сути ранжирование внутри скана, и такие
  признаки в топе по gain.
- Эталон — `wines.label` (OCR‑текст этикетки из каталога) + структурные поля, а не один «good_text»:
  на этикетке и в каталоге разные формулировки, поле `name` часто короче текста на бутылке.
- `--fn-policy skip` по умолчанию: при FN истинное вино неизвестно, размечать нечем.
