# xgboost_text_matcher_abs_v12

FEATURE_VERSION **1.2.0** — 76 абсолютных текстовых признаков.

- без cosine / rel_* / hard_reject / HSV
- омоглифы Vision→кириллица в `normalize_text` (данные в БД не менялись)
- color/type из `wines.label_ocr`
- vintage + `year_match_recency` (не foundation_year)
- порог F1 на маленьком validation ≈ 0.10 — в проде смотрите ещё `text_match_thresholds.xgb`

Данные: `models/data_xgboost_id800_abs_v12`  
Алгоритм пайплайна: **0.69.0**
