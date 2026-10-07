from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str | Path | None = None,
):
    """Plot a confusion matrix without requiring seaborn."""
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "f1_macro",
    save_path: str | Path | None = None,
):
    """Bar chart for one model-comparison metric."""
    plot_df = comparison_df.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(plot_df["model"], plot_df[metric])
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison — {metric}")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)

    for i, value in enumerate(plot_df[metric]):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str | Path | None = None,
):
    """One-vs-rest ROC curves for a multiclass model."""
    num_classes = len(class_names)
    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(num_classes),
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    plotted = 0
    for class_idx, class_name in enumerate(class_names):
        targets = y_true_bin[:, class_idx]

        # ROC is undefined if the test set contains only one binary value.
        if len(np.unique(targets)) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            targets,
            y_prob[:, class_idx],
        )
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC={roc_auc:.3f})",
        )
        plotted += 1

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)

    if plotted:
        ax.legend(loc="lower right")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax
