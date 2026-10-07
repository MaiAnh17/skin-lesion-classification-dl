from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    create_test_loader,
    evaluate_multiple_models,
    load_checkpoint,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_multiclass_roc,
)
from src.models.simple_cnn import SimpleCNN
from src.models.complex_cnn import ComplexCNN
from src.models.transfer_learning import TransferLearningResNet18


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all HAM10000 trained models on the fixed test set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = PROJECT_ROOT / args.config

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    class_to_idx = cfg["classes"]["class_to_idx"]
    class_names = [
        name
        for name, idx in sorted(
            class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]
    num_classes = len(class_names)

    test_csv = PROJECT_ROOT / cfg["data"]["test_csv"]
    raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]

    test_df = pd.read_csv(test_csv)

    test_loader = create_test_loader(
        test_df=test_df,
        data_root=raw_dir,
        class_to_idx=class_to_idx,
        batch_size=int(cfg["data"].get("batch_size", 32)),
        num_workers=int(cfg["data"].get("num_workers", 0)),
        image_size=int(cfg["data"].get("image_size", 224)),
    )

    # IMPORTANT:
    # pretrained=False prevents downloading ImageNet weights during evaluation.
    # The saved project checkpoints replace all model weights immediately.
    models = {
        "Simple CNN": SimpleCNN(
            num_classes=num_classes,
            dropout=0.5,
        ),
        "Complex CNN": ComplexCNN(
            num_classes=num_classes,
        ),
        "ResNet18 Transfer": TransferLearningResNet18(
            num_classes=num_classes,
            dropout=0.3,
            pretrained=False,
            freeze_backbone=True,
        ),
        "ResNet18 Fine-Tuned": TransferLearningResNet18(
            num_classes=num_classes,
            dropout=0.3,
            pretrained=False,
            freeze_backbone=False,
        ),
    }

    checkpoint_map = {
        "Simple CNN": cfg["models"]["simple_cnn"]["checkpoint"],
        "Complex CNN": cfg["models"]["complex_cnn"]["checkpoint"],
        "ResNet18 Transfer": cfg["models"]["transfer_resnet18"]["checkpoint"],
        "ResNet18 Fine-Tuned": cfg["models"]["fine_tuned_resnet18"]["checkpoint"],
    }

    for model_name, relative_path in checkpoint_map.items():
        models[model_name] = load_checkpoint(
            model=models[model_name],
            checkpoint_path=PROJECT_ROOT / relative_path,
            device=device,
        )

    comparison_df, detailed_results = evaluate_multiple_models(
        models=models,
        data_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    metrics_dir = PROJECT_ROOT / cfg["outputs"]["metrics_dir"]
    figures_dir = PROJECT_ROOT / cfg["outputs"]["figures_dir"]
    confusion_dir = PROJECT_ROOT / cfg["outputs"]["confusion_dir"]

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = metrics_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\nFinal comparison:")
    print(comparison_df.to_string(index=False))

    for model_name, result in detailed_results.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        result["classification_report"].to_csv(
            metrics_dir / f"{safe_name}_classification_report.csv"
        )

        predictions_df = test_df[["image_id", "dx"]].copy()
        predictions_df["true_idx"] = result["y_true"]
        predictions_df["pred_idx"] = result["y_pred"]
        predictions_df["pred_dx"] = [
            class_names[i]
            for i in result["y_pred"]
        ]
        predictions_df["correct"] = (
            predictions_df["true_idx"]
            == predictions_df["pred_idx"]
        )
        predictions_df.to_csv(
            metrics_dir / f"{safe_name}_test_predictions.csv",
            index=False,
        )

        with open(
            metrics_dir / f"{safe_name}_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    k: (None if pd.isna(v) else float(v))
                    for k, v in result["metrics"].items()
                },
                f,
                indent=2,
            )

        fig, _ = plot_confusion_matrix(
            result["confusion_matrix"],
            class_names,
            title=f"{model_name} — Confusion Matrix",
            save_path=confusion_dir / f"{safe_name}_confusion_matrix.png",
        )
        fig.clf()

        fig, _ = plot_multiclass_roc(
            result["y_true"],
            result["y_prob"],
            class_names,
            title=f"{model_name} — One-vs-Rest ROC",
            save_path=figures_dir / f"{safe_name}_roc.png",
        )
        fig.clf()

    fig, _ = plot_model_comparison(
        comparison_df,
        metric="f1_macro",
        save_path=figures_dir / "model_comparison_macro_f1.png",
    )
    fig.clf()

    fig, _ = plot_model_comparison(
        comparison_df,
        metric="accuracy",
        save_path=figures_dir / "model_comparison_accuracy.png",
    )
    fig.clf()

    print(f"\nSaved comparison to: {comparison_path}")


if __name__ == "__main__":
    main()
