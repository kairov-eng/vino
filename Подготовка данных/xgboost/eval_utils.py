"""Общие метрики для XGBoost и Cross-Encoder: классификация + ранжирование по скану."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    out: dict[str, float] = {"n": float(len(y)), "positives": float(int(y.sum()))}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    pred = (p >= threshold).astype(int)
    out["threshold"] = float(threshold)
    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["recall"] = float(recall_score(y, pred, zero_division=0))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    neg = y == 0
    out["fpr"] = float(pred[neg].mean()) if neg.any() else 0.0
    return out


def best_f1_threshold(y_true: Sequence[int], y_prob: Sequence[float]) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = float(f), float(t)
    return best_t, best_f1


def topk_accuracy(
    group_ids: Sequence,
    y_true: Sequence[int],
    scores: Sequence[float],
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Top-k по сканам, у которых есть хотя бы один positive.

    Возвращает также MRR и число оценённых сканов.
    """
    by_group: dict = defaultdict(list)
    for g, y, s in zip(group_ids, y_true, scores):
        by_group[g].append((float(s), int(y)))
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    n = 0
    for g, items in by_group.items():
        if not any(y == 1 for _, y in items):
            continue
        n += 1
        items.sort(key=lambda t: -t[0])
        first_pos = next(i for i, (_, y) in enumerate(items) if y == 1)
        rr_sum += 1.0 / (first_pos + 1)
        for k in ks:
            if first_pos < k:
                hits[k] += 1
    out = {f"top{k}_acc": (hits[k] / n if n else 0.0) for k in ks}
    out["mrr"] = rr_sum / n if n else 0.0
    out["scans_with_positive"] = float(n)
    return out


def hard_negative_report(
    kinds: Sequence[str], y_prob: Sequence[float], threshold: float = 0.5
) -> dict[str, dict[str, float]]:
    """Средняя вероятность и доля «пропущенных» по типам примеров (tp/tn/fp_hard...)."""
    by_kind: dict[str, list[float]] = defaultdict(list)
    for k, p in zip(kinds, y_prob):
        by_kind[str(k)].append(float(p))
    out: dict[str, dict[str, float]] = {}
    for k, ps in by_kind.items():
        arr = np.asarray(ps)
        out[k] = {
            "n": float(len(arr)),
            "prob_mean": float(arr.mean()),
            "prob_median": float(np.median(arr)),
            "frac_ge_threshold": float((arr >= threshold).mean()),
        }
    return out


def measure_latency_ms(fn: Callable[[], object], repeats: int = 20, warmup: int = 3) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "repeats": float(repeats),
    }


def print_metrics_table(title: str, blocks: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {title} ===")
    keys: list[str] = []
    for m in blocks.values():
        for k in m:
            if k not in keys:
                keys.append(k)
    width = max(len(k) for k in blocks) + 2
    print(" " * width + "  ".join(f"{k:>14}" for k in keys))
    for name, m in blocks.items():
        cells = []
        for k in keys:
            v = m.get(k)
            cells.append(f"{v:>14.4f}" if isinstance(v, (int, float)) else f"{'-':>14}")
        print(f"{name:<{width}}" + "  ".join(cells))
