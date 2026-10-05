import torch
from PIL import Image

from src.data.preprocessing import (
    get_eval_transform,
)

from src.data.augmentation import (
    get_train_transform,
)


def create_dummy_image():
    return Image.new(
        "RGB",
        (600, 450),
        color=(128, 100, 90),
    )


def test_eval_transform_shape():

    image = create_dummy_image()

    transform = (
        get_eval_transform(
            image_size=224
        )
    )

    tensor = transform(
        image
    )

    assert tensor.shape == (
        3,
        224,
        224,
    )


def test_train_transform_shape():

    image = create_dummy_image()

    transform = (
        get_train_transform(
            image_size=224
        )
    )

    tensor = transform(
        image
    )

    assert tensor.shape == (
        3,
        224,
        224,
    )


def test_eval_transform_deterministic():

    image = create_dummy_image()

    transform = (
        get_eval_transform(
            image_size=224
        )
    )

    tensor1 = transform(
        image
    )

    tensor2 = transform(
        image
    )

    assert torch.equal(
        tensor1,
        tensor2,
    )


def test_tensor_dtype():

    image = create_dummy_image()

    transform = (
        get_eval_transform(
            image_size=224
        )
    )

    tensor = transform(
        image
    )

    assert (
        tensor.dtype
        == torch.float32
    )