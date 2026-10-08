import os
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint_io import read_state_dict
from src.data.augmentation import get_train_transform
from src.data.dataset import HAM10000Dataset, HAM10000_CLASSES
from src.data.preprocessing import get_eval_transform
from src.models import (
    TransferLearningResNet18,
    configure_resnet18_fine_tuning,
    count_trainable_parameters,
    get_fine_tuning_criterion,
    get_fine_tuning_optimizer,
    get_fine_tuning_scheduler,
)
from src.training.losses import compute_class_weights
from src.training.train import evaluate_one_epoch, fit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                return checkpoint[key]
    return checkpoint


def save_training_curves(history: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train")
    ax.plot(history["epoch"], history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Fine-Tuned ResNet-18 - Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fine_tuned_resnet18_loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_accuracy"], label="Train")
    ax.plot(history["epoch"], history["val_accuracy"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fine-Tuned ResNet-18 - Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fine_tuned_resnet18_accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if os.environ.get("HAM10000_ALLOW_TRAINING") != "YES":
        raise RuntimeError("Training disabled to protect existing checkpoints. Run scripts/evaluate.py instead.")
    config_path = PROJECT_ROOT / "configs" / "fine_tuning.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    data_dir = PROJECT_ROOT / "data" / "processed"
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"
    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}. Run scripts/prepare_data.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    image_size = int(cfg["data"].get("image_size", 224))
    batch_size = int(cfg["data"].get("batch_size", 32))
    num_workers = int(cfg["data"].get("num_workers", 0))

    train_dataset = HAM10000Dataset(train_df, PROJECT_ROOT, transform=get_train_transform(image_size=image_size))
    val_dataset = HAM10000Dataset(val_df, PROJECT_ROOT, transform=get_eval_transform(image_size=image_size))
    test_dataset = HAM10000Dataset(test_df, PROJECT_ROOT, transform=get_eval_transform(image_size=image_size))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    transfer_checkpoint = PROJECT_ROOT / cfg["model"]["checkpoint"]
    if not transfer_checkpoint.exists():
        raise FileNotFoundError(f"Transfer-learning checkpoint not found: {transfer_checkpoint}")

    model = TransferLearningResNet18(
        num_classes=int(cfg["data"].get("num_classes", 7)),
        dropout=float(cfg["model"].get("dropout", 0.3)),
        pretrained=False,
        freeze_backbone=False,
        hidden_dim=256,
    )

    checkpoint = read_state_dict(transfer_checkpoint)
    try:
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Transfer checkpoint incompatible with hidden_dim=256. "
            "Use transfer_resnet18_best.pt."
        ) from exc
    model = configure_resnet18_fine_tuning(model).to(device)

    stats = count_trainable_parameters(model)
    print(f"Parameters: total={stats['total']:,}, trainable={stats['trainable']:,}, frozen={stats['frozen']:,}")

    raw_class_weights = compute_class_weights(train_df, HAM10000_CLASSES, device=device)
    criterion, fine_tune_weights = get_fine_tuning_criterion(
        raw_class_weights=raw_class_weights,
        label_smoothing=float(cfg["loss"].get("label_smoothing", 0.05)),
    )
    print(f"Fine-tuning class weights: {fine_tune_weights.detach().cpu().tolist()}")

    optimizer = get_fine_tuning_optimizer(
        model=model,
        backbone_lr=float(cfg["optimizer"].get("backbone_lr", 1e-5)),
        classifier_lr=float(cfg["optimizer"].get("classifier_lr", 1e-4)),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 1e-4)),
    )
    scheduler = get_fine_tuning_scheduler(
        optimizer=optimizer,
        factor=float(cfg["scheduler"].get("factor", 0.5)),
        patience=int(cfg["scheduler"].get("patience", 2)),
        min_lr=float(cfg["scheduler"].get("min_lr", 1e-7)),
    )

    checkpoint_path = PROJECT_ROOT / cfg["outputs"]["checkpoint"]
    history_path = PROJECT_ROOT / cfg["outputs"]["history"]

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=int(cfg["training"].get("epochs", 15)),
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        early_stopping_patience=int(cfg["training"].get("early_stopping_patience", 5)),
        history_path=history_path,
    )

    save_training_curves(history, PROJECT_ROOT / "outputs" / "figures")

    if os.environ.get("HAM10000_TEST_AFTER_TRAINING") != "1":
        print("Final test evaluation deferred to scripts/evaluate.py.")
        return

    best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    model.eval()
    test_loss, test_accuracy = evaluate_one_epoch(model, test_loader, criterion, device)

    print("\nBest fine-tuned ResNet-18 test result")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
