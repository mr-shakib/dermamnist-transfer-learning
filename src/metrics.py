"""Classification metrics used for model selection and the final report."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

from common import CLASS_CODES, NUM_CLASSES


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def summary_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                                         labels=list(range(NUM_CLASSES)))),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "ece": expected_calibration_error(probs, labels),
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()),
    }


def per_class_report(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    aucs = [float(roc_auc_score((labels == k).astype(int), probs[:, k])) for k in range(NUM_CLASSES)]
    return {
        code: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k]),
               "auc": aucs[k]}
        for k, code in enumerate(CLASS_CODES)
    }


def bootstrap_ci(probs: np.ndarray, labels: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile 95% CI for accuracy, balanced accuracy, macro-F1 and macro-AUC (test-set resampling)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")
    draws = {k: [] for k in keys}
    preds = probs.argmax(1)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y, p, pr = labels[idx], preds[idx], probs[idx]
        if len(np.unique(y)) < NUM_CLASSES:
            continue
        draws["accuracy"].append(accuracy_score(y, p))
        draws["balanced_accuracy"].append(balanced_accuracy_score(y, p))
        draws["macro_f1"].append(f1_score(y, p, average="macro", zero_division=0))
        draws["macro_auc"].append(roc_auc_score(y, pr, multi_class="ovr", average="macro"))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def confusion(probs: np.ndarray, labels: np.ndarray) -> list[list[int]]:
    return confusion_matrix(labels, probs.argmax(1), labels=list(range(NUM_CLASSES))).tolist()
