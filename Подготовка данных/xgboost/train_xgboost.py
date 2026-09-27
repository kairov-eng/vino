"""Обучение XGBoost OCR Text Matcher на признаках пар (features/*.parquet).

Вход:  models/data/features/{train,validation,test}.parquet  (из prepare_dataset.py)
Выход: models/xgboost_text_matcher/
         model.json              — модель XGBoost (native JSON, загружается XGBClassifier().load_model)
         feature_names.json      — порядок признаков (обязателен для inference)
         metrics.json            — ROC-AUC / PR-AUC / F1 / Top-k / hard-negatives / latency
         feature_importance.json — gain по признакам
         train_config.json       — гиперпараметры и версия признаков
         predictions_test.csv    — вероятности на test для разбора ошибок

Запуск:
    cd C:\\dev\\Vino2026\\models\\ocr_matcher
    python train_xgboost.py
    python train_xgboost.py --n-estimators 800 --max-depth 4 --learning-rate 0.03
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (  # noqa: E402
    best_f1_threshold,
    classification_metrics,
    hard_negative_report,
    measure_latency_ms,
    print_metrics_table,
    topk_accuracy,
)
from text_features import FEATURE_NAMES, FEATURE_VERSION, WineRef, group_features  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE
DEFAULT_DATA_DIR = MODELS_DIR / "data"
DEFAULT_OUT_DIR = MODELS_DIR / "xgboost_text_matcher"

ID_COLS = ["scan_id", "wine_id", "group_id", "split", "label", "label_kind", "ocr_engine"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--min-child-weight", type=float, default=3.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--early-stopping-rounds", type=int, default=100)
    p.add_argument("--scale-pos-weight", default="1",
                   help="'auto' = neg/pos, число — явно. 1 сохраняет калибровку вероятностей")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-features", default="",
                   help="признаки, которые исключить (через запятую), например для абляции")
    return p.parse_args()


def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / "features" / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(f"Нет файла {path} — сначала запустите prepare_dataset.py")
    return pd.read_parquet(path)


def evaluate(name: str, df: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict:
    y = df["label"].to_numpy(dtype=int)
    m = classification_metrics(y, prob, threshold)
    m.update(topk_accuracy(df["scan_id"].tolist(), y, prob))
    # baseline: чистый SigLIP cosine (если колонка есть — только отчёт)
    if "siglip_cosine" in df.columns:
        cos = df["siglip_cosine"].fillna(-1).to_numpy()
        base = topk_accuracy(df["scan_id"].tolist(), y, cos)
        m["baseline_cosine_top1_acc"] = base["top1_acc"]
        m["baseline_cosine_top3_acc"] = base["top3_acc"]
        m["baseline_cosine_mrr"] = base["mrr"]
    m["hard_negatives"] = hard_negative_report(df["label_kind"].tolist(), prob, threshold)
    return m


def main() -> None:
    args = parse_args()
    train = load_split(args.data_dir, "train")
    valid = load_split(args.data_dir, "validation")
    test = load_split(args.data_dir, "test")

    drop = {x.strip() for x in args.drop_features.split(",") if x.strip()}
    features = [f for f in FEATURE_NAMES if f in train.columns and f not in drop]
    missing = [f for f in FEATURE_NAMES if f not in train.columns]
    if missing:
        print(f"ВНИМАНИЕ: в parquet нет признаков {missing} — пересоберите датасет")
    print(f"train={len(train)} val={len(valid)} test={len(test)} features={len(features)}")

    X_tr, y_tr = train[features], train["label"].astype(int)
    X_va, y_va = valid[features], valid["label"].astype(int)
    X_te, y_te = test[features], test["label"].astype(int)

    sample_weight = None
    if "sample_weight" in train.columns:
        sample_weight = train["sample_weight"].astype(float).to_numpy()
        print(
            f"sample_weight: mean={sample_weight.mean():.3f} "
            f"max={sample_weight.max():.2f} "
            f"rows_gt1={(sample_weight > 1.0).sum()}"
        )

    if str(args.scale_pos_weight).lower() == "auto":
        spw = float((y_tr == 0).sum()) / max(1, int((y_tr == 1).sum()))
    else:
        spw = float(args.scale_pos_weight)
    print(f"scale_pos_weight={spw:.3f}  positives train={int(y_tr.sum())} / {len(y_tr)}")

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric=["logloss", "aucpr"],
        tree_method="hist",
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=0,
    )
    fit_kwargs: dict = {"eval_set": [(X_va, y_va)], "verbose": 100}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_tr, y_tr, **fit_kwargs)
    best_it = int(getattr(model, "best_iteration", model.n_estimators) or model.n_estimators)
    print(f"best_iteration={best_it}")

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    p_tr = model.predict_proba(X_tr)[:, 1]

    thr, f1_va = best_f1_threshold(y_va, p_va)
    print(f"best F1 threshold (validation) = {thr:.2f}  F1={f1_va:.4f}")

    metrics = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "best_iteration": best_it,
        "threshold_best_f1_validation": thr,
        "train": evaluate("train", train, p_tr, thr),
        "validation": evaluate("validation", valid, p_va, thr),
        "test": evaluate("test", test, p_te, thr),
        "test_at_0_5": classification_metrics(y_te, p_te, 0.5),
    }

    # --- latency: 20 кандидатов (только predict) и (features + predict) ---
    wines = {}
    wines_path = args.data_dir / "raw" / "wines.jsonl"
    if wines_path.exists():
        for line in wines_path.open(encoding="utf-8"):
            w = json.loads(line)
            wines[w["wine_id"]] = w
    sample_scan = test["scan_id"].iloc[0]
    sample = test[test["scan_id"] == sample_scan]
    X_20 = sample[features].head(20)
    lat_predict = measure_latency_ms(lambda: model.predict_proba(X_20), repeats=50)
    refs = [
        WineRef(wine_id=int(w), label=(wines.get(int(w)) or {}).get("label"),
                name=(wines.get(int(w)) or {}).get("name"), winery=(wines.get(int(w)) or {}).get("winery"),
                category=(wines.get(int(w)) or {}).get("category"), grape=(wines.get(int(w)) or {}).get("grape"),
                region=(wines.get(int(w)) or {}).get("region"))
        for w in sample["wine_id"].head(20)
    ]
    cosines = (
        sample["siglip_cosine"].head(20).tolist()
        if "siglip_cosine" in sample.columns
        else [None] * min(20, len(sample))
    )
    ocr_sample = "Усадьба Перовских\nКРЫМСКИЙ БЛЕНД\n2023\nКРЫМ\nСевастополь\n0,75 л"

    def _full() -> None:
        rows = group_features(ocr_sample, refs, cosines)
        # Для чистого latency-бенчмарка hard-reject признаки не вычисляются:
        # добавляем их нулями, сохраняя тот же размер входа модели.
        frame = pd.DataFrame(rows).reindex(columns=features, fill_value=0.0)
        model.predict_proba(frame)

    lat_full = measure_latency_ms(_full, repeats=20)
    metrics["latency"] = {
        "predict_only_20_candidates": lat_predict,
        "features_plus_predict_20_candidates": lat_full,
        "device": "cpu",
    }

    # --- вывод ---
    print_metrics_table(
        "XGBoost — классификация (threshold = best F1 on validation)",
        {
            sp: {k: v for k, v in metrics[sp].items() if k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "fpr")}
            for sp in ("train", "validation", "test")
        },
    )
    print_metrics_table(
        "XGBoost — ранжирование по скану (Top-k) vs baseline SigLIP cosine",
        {
            sp: {k: v for k, v in metrics[sp].items()
                 if k in ("top1_acc", "top3_acc", "top5_acc", "mrr", "baseline_cosine_top1_acc", "baseline_cosine_top3_acc", "scans_with_positive")}
            for sp in ("train", "validation", "test")
        },
    )
    fp_reports = {
        kind: report
        for kind, report in metrics["test"]["hard_negatives"].items()
        if "fp_hard" in kind
    }
    print(
        "\nHard negatives (FP-разметка) на test:",
        json.dumps(fp_reports or None, ensure_ascii=False),
    )
    print(f"Latency: predict 20 cand = {lat_predict['median_ms']:.2f} ms; features+predict 20 cand = {lat_full['median_ms']:.2f} ms")

    # --- сохранение ---
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "model.json")
    (out / "feature_names.json").write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    imp = model.get_booster().get_score(importance_type="gain")
    imp_sorted = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    (out / "feature_importance.json").write_text(json.dumps(imp_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update({
        "scale_pos_weight_effective": spw,
        "feature_version": FEATURE_VERSION,
        "xgboost_version": __import__("xgboost").__version__,
        "n_features": len(features),
    })
    (out / "train_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_cols = [c for c in ID_COLS + ["siglip_cosine"] if c in test.columns]
    pred = test[pred_cols].copy()
    pred["prob"] = p_te
    pred.sort_values(["scan_id", "prob"], ascending=[True, False]).to_csv(out / "predictions_test.csv", index=False)

    print("\nTop-15 признаков по gain:")
    for k, v in list(imp_sorted.items())[:15]:
        print(f"  {k:<32} {v:.3f}")
    print(f"\nМодель сохранена: {out / 'model.json'}")


if __name__ == "__main__":
    main()
