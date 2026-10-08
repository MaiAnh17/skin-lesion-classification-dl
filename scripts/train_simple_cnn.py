import os
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import get_train_transform
from src.data.dataset import HAM10000Dataset, HAM10000_CLASSES
from src.data.preprocessing import get_eval_transform
from src.models import SimpleCNN
from src.training.losses import compute_class_weights, get_weighted_cross_entropy_loss
from src.training.train import evaluate_one_epoch, fit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_training_curves(history: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train")
    ax.plot(history["epoch"], history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Simple CNN - Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "simple_cnn_loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_accuracy"], label="Train")
    ax.plot(history["epoch"], history["val_accuracy"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Simple CNN - Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "simple_cnn_accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if os.environ.get("HAM10000_ALLOW_TRAINING") != "YES":
        raise RuntimeError("Training disabled to protect existing checkpoints. Run scripts/evaluate.py instead.")
    config_path = PROJECT_ROOT / "configs" / "simple_cnn.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    seed = int(cfg.get("seed", 42))
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

    model = SimpleCNN(
        num_classes=int(cfg["model"].get("num_classes", 7)),
        dropout=float(cfg["model"].get("dropout", 0.5)),
    ).to(device)

    class_weights = compute_class_weights(train_df, HAM10000_CLASSES, device=device)
    criterion = get_weighted_cross_entropy_loss(
        class_weights,
        label_smoothing=float(cfg.get("loss", {}).get("label_smoothing", 0.0)),
        soften=bool(cfg.get("loss", {}).get("soften_class_weights", False)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    scheduler = None
    scheduler_cfg = cfg.get("scheduler", {})
    if scheduler_cfg.get("use_scheduler", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 3)),
        )

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "simple_cnn_best.pt"
    history_path = PROJECT_ROOT / "outputs" / "metrics" / "simple_cnn_training_history.csv"

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=int(cfg["training"].get("epochs", 30)),
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        early_stopping_patience=int(cfg.get("early_stopping", {}).get("patience", 7)),
        history_path=history_path,
    )

    save_training_curves(history, PROJECT_ROOT / "outputs" / "figures")

    if os.environ.get("HAM10000_TEST_AFTER_TRAINING") != "1":
        print("Final test evaluation deferred to scripts/evaluate.py.")
        return

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    test_loss, test_accuracy = evaluate_one_epoch(model, test_loader, criterion, device)
    print("\nBest Simple CNN test result")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
