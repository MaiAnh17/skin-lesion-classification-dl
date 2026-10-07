from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Compute global classification metrics for a multiclass problem."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }

    # Multiclass ROC-AUC can fail if a class is absent from y_true.
    try:
        labels = np.arange(len(class_names))
        metrics["roc_auc_macro_ovr"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        metrics["roc_auc_weighted_ovr"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        metrics["roc_auc_macro_ovr"] = np.nan
        metrics["roc_auc_weighted_ovr"] = np.nan

    return metrics


def classification_report_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """Return sklearn classification_report as a DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def confusion_matrix_array(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return a fixed-size confusion matrix."""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
    )
