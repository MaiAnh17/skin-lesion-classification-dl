from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.data.dataset import (
    HAM10000Dataset,
)

from src.data.preprocessing import (
    get_eval_transform,
)


def test_dataset_length(
    tmp_path,
):

    image_path = (
        tmp_path
        / "sample.jpg"
    )

    image = Image.new(
        "RGB",
        (600, 450),
    )

    image.save(
        image_path
    )

    df = pd.DataFrame(
        {
            "image_id": [
                "sample"
            ],
            "lesion_id": [
                "lesion_1"
            ],
            "dx": [
                "nv"
            ],
            "image_path": [
                str(image_path)
            ],
        }
    )

    dataset = HAM10000Dataset(
        dataframe=df,
        project_root=tmp_path,
        transform=get_eval_transform(),
    )

    assert len(dataset) == 1


def test_dataset_output(
    tmp_path,
):

    image_path = (
        tmp_path
        / "sample.jpg"
    )

    image = Image.new(
        "RGB",
        (600, 450),
    )

    image.save(
        image_path
    )

    df = pd.DataFrame(
        {
            "image_id": [
                "sample"
            ],
            "lesion_id": [
                "lesion_1"
            ],
            "dx": [
                "mel"
            ],
            "image_path": [
                str(image_path)
            ],
        }
    )

    dataset = HAM10000Dataset(
        dataframe=df,
        project_root=tmp_path,
        transform=get_eval_transform(),
    )

    image_tensor, label = (
        dataset[0]
    )

    assert (
        image_tensor.shape
        == (
            3,
            224,
            224,
        )
    )

    assert isinstance(
        label,
        torch.Tensor,
    )

    assert (
        label.dtype
        == torch.long
    )