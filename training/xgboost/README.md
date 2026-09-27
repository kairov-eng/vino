# XGBoost OCR↔label matcher — обучение

Прод-модель в git: `backend/models/xgboost_text_matcher_abs_v14/`  
(`model.json` + `feature_names.json` + метрики).

## Архив с данными и скриптами

Файл: **`xgboost_train_bundle.zip`** (в этой папке)

Содержимое архива:

```
ocr_matcher/          # скрипты подготовки и train
  prepare_dataset.py
  prepare_xgboost_dataset_query_ocr.py
  prepare_xgboost_synthetic.py
  train_xgboost.py
  text_features.py
  ...
data_xgboost_id800_abs_v14/   # готовый датасет abs_v14
  features/{train,validation,test}.parquet
  raw/{groups,wines}.jsonl
  dataset_stats.json
```

## Как переобучить

```powershell
# 1) распаковать архив
Expand-Archive training\xgboost\xgboost_train_bundle.zip -DestinationPath training\xgboost\work -Force
cd training\xgboost\work\ocr_matcher

# 2) обучение на готовых parquet
python train_xgboost.py `
  --data-dir ..\data_xgboost_id800_abs_v14 `
  --out-dir ..\..\..\backend\models\xgboost_text_matcher_abs_v14_new

# 3) пересобрать датасет из БД (нужен DATABASE_URL + search_photos):
python prepare_xgboost_dataset_query_ocr.py --min-scan-id 800 --out-dir ..\data_xgboost_new
python train_xgboost.py --data-dir ..\data_xgboost_new --out-dir ...
```

Подробнее: исходный README в `research/models/ocr_matcher/README.md` (локальный monorepo).
