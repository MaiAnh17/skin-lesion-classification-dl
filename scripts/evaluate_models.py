import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch_directml
from src.data.dataset import HAM10000Dataset, HAM10000_CLASSES
from src.data.preprocessing import get_eval_transform
from src.models.simple_cnn import SimpleCNN
from src.models.complex_cnn import ComplexCNN
from src.models.transfer_learning import TransferLearningResNet18


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    return acc, macro_f1, weighted_f1, all_targets, all_preds


def main():
    torch.set_num_threads(14)
    device = torch_directml.device()
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    eval_transform = get_eval_transform(image_size=224)
    test_dataset = HAM10000Dataset(test_df, PROJECT_ROOT, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"

    results = []

    # 1. Simple CNN
    p = checkpoints_dir / "simple_cnn_best.pt"
    if p.exists():
        m = SimpleCNN(num_classes=7)
        state = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(state)
        m.to(device)
        acc, mf1, wf1, _, _ = evaluate(m, test_loader, device)
        results.append({"Model": "Simple CNN", "Accuracy": acc, "Macro F1": mf1, "Weighted F1": wf1})

    # 2. Refined Complex CNN
    p = checkpoints_dir / "complex_cnn_refined_best.pt"
    if p.exists():
        m = ComplexCNN(num_classes=7)
        state = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(state)
        m.to(device)
        acc, mf1, wf1, _, _ = evaluate(m, test_loader, device)
        results.append({"Model": "Complex CNN (Refined)", "Accuracy": acc, "Macro F1": mf1, "Weighted F1": wf1})

    # 3. Refined Transfer Learning
    p = checkpoints_dir / "transfer_resnet18_refined_best.pt"
    if p.exists():
        m = TransferLearningResNet18(num_classes=7, hidden_dim=256)
        state = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(state)
        m.to(device)
        acc, mf1, wf1, _, _ = evaluate(m, test_loader, device)
        results.append({"Model": "Transfer Learning (Refined)", "Accuracy": acc, "Macro F1": mf1, "Weighted F1": wf1})

    # 4. Fine-Tuned ResNet-18
    p = checkpoints_dir / "fine_tuned_resnet18_best.pt"
    if p.exists():
        m = TransferLearningResNet18(num_classes=7, hidden_dim=None)
        state = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(state)
        m.to(device)
        acc, mf1, wf1, targets, preds = evaluate(m, test_loader, device)
        results.append({"Model": "Fine-Tuning ResNet-18", "Accuracy": acc, "Macro F1": mf1, "Weighted F1": wf1})

    df = pd.DataFrame(results)
    print("\n--- Final Test Set Summary Across All Models ---", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
