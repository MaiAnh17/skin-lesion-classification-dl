from typing import Sequence

from torchvision import transforms

from src.data.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_train_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
):
    """
    Training preprocessing and augmentation.
    """

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.85, 1.0),
                ratio=(0.90, 1.10),
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomVerticalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=20
            ),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )