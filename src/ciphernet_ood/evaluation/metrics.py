from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    labels = sorted(set(y_true) | set(y_pred))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "g_mean": float(np.prod(np.clip(recalls, 1e-12, 1.0)) ** (1.0 / max(len(recalls), 1))),
    }


def imbalance_metrics(y_true, y_pred) -> dict[str, float]:
    labels, counts = np.unique(y_true, return_counts=True)
    threshold = np.median(counts)
    minority = labels[counts <= threshold]
    majority = labels[counts > threshold]
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_by_label = dict(zip(labels, per_class_f1))
    f1_min = float(np.mean([f1_by_label[label] for label in minority])) if len(minority) else 0.0
    f1_maj = float(np.mean([f1_by_label[label] for label in majority])) if len(majority) else f1_min
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {"minority_f1": f1_min, "majority_f1": f1_maj, "f1_gap": f1_maj - f1_min, "worst_class_recall": float(np.min(recalls))}


def ood_metrics(is_ood_true, scores, threshold: float) -> dict[str, float]:
    is_ood_true = np.asarray(is_ood_true).astype(int)
    scores = np.asarray(scores)
    pred = (scores > threshold).astype(int)
    fpr, tpr, thresholds = roc_curve(is_ood_true, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return {
        "auroc": float(roc_auc_score(is_ood_true, scores)),
        "aupr_ood": float(average_precision_score(is_ood_true, scores)),
        "fpr_at_95_tpr": float(fpr[idx]),
        "f1_ood": float(f1_score(is_ood_true, pred, zero_division=0)),
    }


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).tolist()
