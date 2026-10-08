import os
import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch_directml
from src.data.dataset import HAM10000Dataset, HAM10000_CLASSES
from src.data.preprocessing import get_eval_transform
from src.data.augmentation import get_train_transform
from src.checkpoint_io import read_state_dict
from src.models.transfer_learning import TransferLearningResNet18
from src.models.fine_tuning import configure_resnet18_fine_tuning, count_trainable_parameters
from src.training.losses import compute_class_weights, get_weighted_cross_entropy_loss
from src.training.train import fit, evaluate_one_epoch


def main():
    if os.environ.get("HAM10000_ALLOW_TRAINING") != "YES":
        raise RuntimeError("Training disabled to protect existing checkpoints. Run scripts/evaluate.py instead.")
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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 1. Initialize TransferLearningResNet18 with refined 2-layer head
    model = TransferLearningResNet18(
        num_classes=7,
        dropout=0.3,
        pretrained=False,
        freeze_backbone=True,
        hidden_dim=256,
    )

    # 2. Load the best refined Transfer Learning checkpoint
    transfer_ckpt_path = checkpoints_dir / "transfer_resnet18_best.pt"
    if not transfer_ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing refined transfer checkpoint: {transfer_ckpt_path}. "
            "Do not substitute the original one-layer classifier checkpoint."
        )
    
    print(f"Loading base transfer weights from: {transfer_ckpt_path.name}...", flush=True)
    state = read_state_dict(transfer_ckpt_path)
    model.load_state_dict(state)
    print("Base transfer weights loaded successfully.", flush=True)

    # 3. Configure progressive fine-tuning (unfreeze layer4 + fc)
    model = configure_resnet18_fine_tuning(model).to(dml)
    stats = count_trainable_parameters(model)
    print(
        f"Parameters: Total={stats['total']:,} | "
        f"Trainable={stats['trainable']:,} ({stats['trainable_ratio']*100:.2f}%) | "
        f"Frozen={stats['frozen']:,}",
        flush=True,
    )

    # 4. Softened class weights matching fine-tuning spec
    raw_weights = compute_class_weights(train_df, HAM10000_CLASSES, device=dml)
    criterion = get_weighted_cross_entropy_loss(raw_weights, label_smoothing=0.05, soften=True)

    # 5. Discriminative AdamW optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.layer4.parameters(), "lr": 1e-5},
            {"params": model.backbone.fc.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    checkpoint_path = checkpoints_dir / "fine_tuned_resnet18_best.pt"
    history_path = metrics_dir / "fine_tuned_resnet18_refined_history.csv"

    print("\nStarting Fine-Tuning Training for ResNet-18 (Layer4 + 2-layer FC)...", flush=True)
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=dml,
        num_epochs=15,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        early_stopping_patience=5,
        history_path=history_path,
    )

    if os.environ.get("HAM10000_TEST_AFTER_TRAINING") != "1":
        print("Final test evaluation deferred to scripts/evaluate.py.")
        return

    print("\nLoading best fine-tuned model for test evaluation...", flush=True)
    best_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_state)
    model.to(dml)

    test_loss, test_acc = evaluate_one_epoch(model, test_loader, criterion, dml)

    # Detailed metrics
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(dml)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    print("\n==========================================", flush=True)
    print("Final Fine-Tuning Test Set Evaluation:", flush=True)
    print(f"Test Loss:        {test_loss:.4f}", flush=True)
    print(f"Test Accuracy:    {test_acc:.4f} ({test_acc * 100:.2f}%)", flush=True)
    print(f"Test Macro F1:    {macro_f1:.4f}", flush=True)
    print(f"Test Weighted F1: {weighted_f1:.4f}", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()
