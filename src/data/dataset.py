from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


HAM10000_CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


CLASS_TO_IDX = {
    class_name: idx
    for idx, class_name in enumerate(HAM10000_CLASSES)
}


IDX_TO_CLASS = {
    idx: class_name
    for class_name, idx in CLASS_TO_IDX.items()
}


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.

    Expected metadata columns:
        image_id
        lesion_id
        dx
        image_path
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        project_root: Path,
        transform: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        dataframe:
            Metadata DataFrame.

        project_root:
            Root directory of the project.

        transform:
            torchvision transformation pipeline.
        """

        self.df = dataframe.reset_index(
            drop=True
        ).copy()

        self.project_root = Path(
            project_root
        )

        self.transform = transform

        self.class_to_idx = CLASS_TO_IDX
        self.idx_to_class = IDX_TO_CLASS

        self._validate_dataframe()

    def _validate_dataframe(self):
        """
        Validate required columns and labels.
        """

        required_columns = {
            "image_id",
            "lesion_id",
            "dx",
            "image_path",
        }

        missing_columns = (
            required_columns
            - set(self.df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        invalid_labels = (
            set(self.df["dx"].unique())
            - set(HAM10000_CLASSES)
        )

        if invalid_labels:
            raise ValueError(
                f"Unknown diagnosis labels: "
                f"{sorted(invalid_labels)}"
            )

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, stored_path, image_id=None):
        """Resolve existing paths or find images after moving the project.

        Does not rewrite split CSVs or change their row order/labels.
        """
        if pd.notna(stored_path) and str(stored_path).strip():
            raw = Path(str(stored_path))
            candidates = [raw]
            if not raw.is_absolute():
                candidates.extend([
                    self.project_root / raw,
                    self.project_root / "data" / "raw" / raw,
                ])
            for candidate in candidates:
                if candidate.is_file():
                    return candidate

        if image_id is None:
            raise FileNotFoundError(f"Image path no longer exists: {stored_path}")
        image_id = str(image_id).strip()
        raw_dir = self.project_root / "data" / "raw"
        for base in (raw_dir / "HAM10000_images_part_1",
                     raw_dir / "HAM10000_images_part_2", raw_dir):
            for extension in (".jpg", ".jpeg", ".png"):
                candidate = base / f"{image_id}{extension}"
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(
            f"Image {image_id} not found. Checked stored_path={stored_path!r} "
            f"and standard HAM10000 folders in {raw_dir}. "
            "Keep the existing CSV and place images in data/raw."
        )

    def __getitem__(
        self,
        index,
    ):
        row = self.df.iloc[index]

        image_path = (
            self._resolve_image_path(
                row["image_path"], image_id=row["image_id"]
            )
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        # -----------------------------------------------------
        # Load image
        # -----------------------------------------------------

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            if self.transform:
                image = self.transform(
                    image
                )

        # -----------------------------------------------------
        # Encode label
        # -----------------------------------------------------

        label_name = row["dx"]

        label = self.class_to_idx[
            label_name
        ]

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image, label

def get_class_mapping():
    """
    Return diagnosis label mappings.
    """

    return (
        CLASS_TO_IDX.copy(),
        IDX_TO_CLASS.copy(),
    )