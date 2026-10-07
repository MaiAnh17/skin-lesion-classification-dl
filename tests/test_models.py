import torch

from src.models import (
    ComplexCNN,
    SimpleCNN,
    TransferLearningResNet18,
    configure_resnet18_fine_tuning,
)


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=7)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        output = model(x)
    assert output.shape == (2, 7)


def test_complex_cnn_output_shape():
    model = ComplexCNN(num_classes=7)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        output = model(x)
    assert output.shape == (2, 7)


def test_transfer_learning_output_shape_without_download():
    model = TransferLearningResNet18(num_classes=7, pretrained=False, freeze_backbone=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        output = model(x)
    assert output.shape == (1, 7)


def test_transfer_learning_freezes_backbone_but_not_classifier():
    model = TransferLearningResNet18(num_classes=7, pretrained=False, freeze_backbone=True)
    assert not any(
        parameter.requires_grad
        for name, parameter in model.backbone.named_parameters()
        if not name.startswith("fc.")
    )
    assert all(parameter.requires_grad for parameter in model.backbone.fc.parameters())


def test_fine_tuning_unfreezes_only_layer4_and_classifier():
    model = TransferLearningResNet18(num_classes=7, pretrained=False, freeze_backbone=False)
    configure_resnet18_fine_tuning(model)

    assert all(parameter.requires_grad for parameter in model.backbone.layer4.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.fc.parameters())

    for module_name in ("conv1", "bn1", "layer1", "layer2", "layer3"):
        module = getattr(model.backbone, module_name)
        assert not any(parameter.requires_grad for parameter in module.parameters())
