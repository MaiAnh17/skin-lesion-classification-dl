import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch_directml
from src.data.dataset import HAM10000Dataset, HAM10000_CLASSES
from src.data.preprocessing import get_eval_transform
from src.data.augmentation import get_train_transform
from src.models.transfer_learning import TransferLearningResNet18
from src.training.losses import compute_class_weights, get_weighted_cross_entropy_loss
from src.training.train import fit, evaluate_one_epoch

def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(14)

    dml = torch_directml.device()
    gpu_name = torch_directml.device_name(0)
    print(f"Device: DirectML on {gpu_name}", flush=True)
    print(f"Host CPU threads: {torch.get_num_threads()}", flush=True)

    data_dir = PROJECT_ROOT / "data" / "processed"
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}", flush=True)

    train_transform = get_train_transform(image_size=224)
    eval_transform = get_eval_transform(image_size=224)

    train_dataset = HAM10000Dataset(train_df, PROJECT_ROOT, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, PROJECT_ROOT, transform=eval_transform)
    test_dataset = HAM10000Dataset(test_df, PROJECT_ROOT, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    # Transfer Learning ResNet-18 (Backbone frozen, trainable linear probe)
    model = TransferLearningResNet18(
        num_classes=7,
        dropout=0.3,
        pretrained=True,
        freeze_backbone=True,
        hidden_dim=256,
    ).to(dml)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}", flush=True)

    # Softened class weights matching fine-tuning
    raw_weights = compute_class_weights(train_df, HAM10000_CLASSES, device=dml)
    criterion = get_weighted_cross_entropy_loss(raw_weights, label_smoothing=0.05, soften=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    checkpoint_path = checkpoints_dir / "transfer_resnet18_refined_best.pt"
    history_path = metrics_dir / "transfer_resnet18_refined_history.csv"

    print("\nStarting training for Refined Transfer Learning ResNet-18...", flush=True)
    history_df = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=dml,
        num_epochs=20,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        early_stopping_patience=6,
        history_path=history_path,
    )

    print("\nLoading best model for test evaluation...", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=dml, weights_only=False))
    test_loss, test_acc = evaluate_one_epoch(model, test_loader, criterion, dml)
    print(f"\nFinal Test Evaluation:", flush=True)
    print(f"Test Loss: {test_loss:.4f}", flush=True)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)", flush=True)

if __name__ == "__main__":
    main()
