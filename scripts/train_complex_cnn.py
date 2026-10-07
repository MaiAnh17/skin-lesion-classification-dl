import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch_directml
from src.data.dataset import HAM10000Dataset, HAM10000_CLASSES
from src.data.preprocessing import get_eval_transform
from src.data.augmentation import get_train_transform
from src.models.complex_cnn import ComplexCNN
from src.training.losses import compute_class_weights, get_weighted_cross_entropy_loss
from src.training.train import fit, evaluate_one_epoch

def main():
    # 1. Reproducibility & Threads
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(14)

    # 2. Device: DirectML (RX 6700 XT)
    dml = torch_directml.device()
    gpu_name = torch_directml.device_name(0)
    print(f"Device: DirectML on {gpu_name}")
    print(f"CPU threads for data/host: {torch.get_num_threads()}")

    # 3. Paths
    data_dir = PROJECT_ROOT / "data" / "processed"
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"Splits loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 4. Transforms and DataLoaders
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

    # 5. Model: Refined Complex CNN (Fix A)
    # - Removed cascading Dropout2d from conv blocks
    # - Pool size changed to (4, 4) to preserve spatial features
    # - Dropout in FC head set to 0.4
    model = ComplexCNN(
        num_classes=7,
        dropout=0.4,
        block_dropout=[0.0, 0.0, 0.0, 0.0],
        adaptive_pool_size=(4, 4),
    ).to(dml)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 6. Loss: Smoothed inverse class weights + label smoothing
    raw_weights = compute_class_weights(train_df, HAM10000_CLASSES, device=dml)
    criterion = get_weighted_cross_entropy_loss(raw_weights, label_smoothing=0.05, soften=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    checkpoint_path = checkpoints_dir / "complex_cnn_refined_best.pt"
    history_path = metrics_dir / "complex_cnn_refined_history.csv"

    print("\nStarting training for Refined Complex CNN (Fix A)...")
    history_df = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=dml,
        num_epochs=25,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        early_stopping_patience=7,
        history_path=history_path,
    )

    # 7. Evaluate Best Model on Test Set
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=dml))
    test_loss, test_acc = evaluate_one_epoch(model, test_loader, criterion, dml)
    print(f"\nFinal Test Evaluation:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

if __name__ == "__main__":
    main()
