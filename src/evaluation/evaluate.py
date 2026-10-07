from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .metrics import (
    classification_report_dataframe,
    compute_classification_metrics,
    confusion_matrix_array,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_evaluation_transform(image_size: int = 224):
    """
    Deterministic transform for validation/test data.

    IMPORTANT:
    - No random augmentation.
    - Uses ImageNet normalization, matching Parts 4–8.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class HAM10000EvaluationDataset(Dataset):
    """
    Evaluation-only HAM10000 dataset.

    Expected metadata columns:
    - image_id
    - dx

    Image path resolution supports:
    1) an explicit `image_path` column, or
    2) the standard HAM10000 folders:
       data/raw/HAM10000_images_part_1
       data/raw/HAM10000_images_part_2
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_root: str | Path,
        class_to_idx: Dict[str, int],
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform or get_evaluation_transform()

        required = {"image_id", "dx"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in metadata: {sorted(missing)}")

        unknown = set(self.df["dx"].dropna().unique()) - set(self.class_to_idx)
        if unknown:
            raise ValueError(f"Unknown dx labels found: {sorted(unknown)}")

        self.part1 = self.data_root / "HAM10000_images_part_1"
        self.part2 = self.data_root / "HAM10000_images_part_2"

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        if "image_path" in self.df.columns and pd.notna(row.get("image_path")):
            p = Path(str(row["image_path"]))
            if not p.is_absolute():
                p = self.data_root / p
            if p.exists():
                return p

        image_id = str(row["image_id"])
        candidates = [
            self.part1 / f"{image_id}.jpg",
            self.part2 / f"{image_id}.jpg",
            self.part1 / f"{image_id}.jpeg",
            self.part2 / f"{image_id}.jpeg",
            self.part1 / f"{image_id}.png",
            self.part2 / f"{image_id}.png",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Image not found for image_id={image_id}. "
            f"Checked standard HAM10000 folders under: {self.data_root}"
        )

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._resolve_image_path(row)

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        label = self.class_to_idx[str(row["dx"])]
        return image, label


def create_test_loader(
    test_df: pd.DataFrame,
    data_root: str | Path,
    class_to_idx: Dict[str, int],
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
) -> DataLoader:
    """Create the single fixed test DataLoader used by all models."""
    dataset = HAM10000EvaluationDataset(
        dataframe=test_df,
        data_root=data_root,
        class_to_idx=class_to_idx,
        transform=get_evaluation_transform(image_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _extract_state_dict(checkpoint):
    """
    Support either:
    - raw model state_dict
    - {'model_state_dict': ...}
    - {'state_dict': ...}
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = True,
) -> torch.nn.Module:
    """Load a saved best checkpoint without retraining."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    state_dict = _extract_state_dict(checkpoint)

    # Support checkpoints saved from DataParallel.
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference.

    Returns
    -------
    y_true : (N,)
    y_pred : (N,)
    y_prob : (N, C)
    """
    all_true = []
    all_pred = []
    all_prob = []

    model.eval()

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_true.append(labels.cpu().numpy())
        all_pred.append(predictions.cpu().numpy())
        all_prob.append(probabilities.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, object]:
    """Evaluate one trained model on the fixed test set."""
    y_true, y_pred, y_prob = predict(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
    )

    report_df = classification_report_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    cm = confusion_matrix_array(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=len(class_names),
    )

    return {
        "metrics": metrics,
        "classification_report": report_df,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate_multiple_models(
    models: Dict[str, torch.nn.Module],
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    Evaluate every model on exactly the same test loader.

    Returns
    -------
    comparison_df
        One row per model.
    detailed_results
        Full predictions/reports/confusion matrices for each model.
    """
    rows = []
    detailed_results = {}

    for model_name, model in models.items():
        print(f"Evaluating: {model_name}")

        result = evaluate_model(
            model=model,
            data_loader=data_loader,
            device=device,
            class_names=class_names,
        )

        detailed_results[model_name] = result

        row = {"model": model_name}
        row.update(result["metrics"])
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    if "f1_macro" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(
            "f1_macro",
            ascending=False,
        ).reset_index(drop=True)

    return comparison_df, detailed_results
