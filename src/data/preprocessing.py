from typing import Sequence

from torchvision import transforms


DEFAULT_IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def get_base_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
):
    """
    Basic preprocessing before tensor conversion.

    Used as a reusable base transform.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
        ]
    )


def get_eval_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
):
    """
    Validation / test preprocessing.

    No random augmentation.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def denormalize_tensor(
    tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
):
    """
    Undo normalization for visualization.
    """

    tensor = tensor.clone()

    for channel, channel_mean, channel_std in zip(
        tensor,
        mean,
        std,
    ):
        channel.mul_(
            channel_std
        )

        channel.add_(
            channel_mean
        )

    return tensor.clamp(
        0,
        1,
    )