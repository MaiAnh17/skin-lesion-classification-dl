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

try:
    import torch_directml
except ImportError:
    torch_directml = None
from src.data.dataset import HAM10000Dataset, HAM10000_CLASSES
from src.data.preprocessing import get_eval_transform
from src.checkpoint_io import load_model
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
    device = torch_directml.device() if torch_directml is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Evaluation device: {device}")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    eval_transform = get_eval_transform(image_size=224)
    test_dataset = HAM10000Dataset(test_df, PROJECT_ROOT, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"

    results = []
    missing_checkpoints = []
    expected = [
        "simple_cnn_best.pt", "complex_cnn_best.pt",
        "transfer_resnet18_best.pt", "fine_tuned_resnet18_best.pt",
    ]
    for filename in expected:
        if not (checkpoints_dir / filename).is_file():
            missing_checkpoints.append(filename)
    if missing_checkpoints:
        print("WARNING: Checkpoints not found (these models will be omitted):")
        for filename in missing_checkpoints:
            print(f"  - {filename}")


    specs = [
        ("Simple CNN", "simple_cnn_best.pt", SimpleCNN(num_classes=7)),
        ("Complex CNN", "complex_cnn_best.pt", ComplexCNN(num_classes=7)),
        ("Transfer Learning ResNet18", "transfer_resnet18_best.pt",
         TransferLearningResNet18(num_classes=7, dropout=0.3, pretrained=False, hidden_dim=256)),
        ("Fine-Tuned ResNet18", "fine_tuned_resnet18_best.pt",
         TransferLearningResNet18(num_classes=7, dropout=0.3, pretrained=False, hidden_dim=256)),
    ]
    for name, filename, model in specs:
        path = checkpoints_dir / filename
        if not path.is_file():
            continue
        model = load_model(model, path, device)
        acc, mf1, wf1, _, _ = evaluate(model, test_loader, device)
        results.append({"Model": name, "Accuracy": acc, "Macro F1": mf1, "Weighted F1": wf1})

    if not results:
        raise FileNotFoundError(f"No compatible checkpoints found under {checkpoints_dir}")
    df = pd.DataFrame(results)
    out_dir = PROJECT_ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "model_comparison_refined.csv"
    df.to_csv(out_file, index=False)
    print("\n--- Final Test Set Summary Across All Models ---", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"Saved comparison to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
