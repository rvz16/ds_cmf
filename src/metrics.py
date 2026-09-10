"""Prediction and trading metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


LABELS = [-1, 0, 1]


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list]:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "precision_down": float(precision[0]),
        "precision_neutral": float(precision[1]),
        "precision_up": float(precision[2]),
        "recall_down": float(recall[0]),
        "recall_neutral": float(recall[1]),
        "recall_up": float(recall[2]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
