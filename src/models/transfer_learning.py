from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class TransferLearningResNet18(nn.Module):
    """ResNet-18 transfer-learning model for HAM10000 (7 classes).

    The pretrained convolutional backbone is frozen by default and only the
    new classification head is optimized during the transfer-learning stage.
    Fine-tuning can be performed later by unfreezing selected backbone blocks.
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features  # 512

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze every pretrained layer and keep the new classifier trainable."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def unfreeze_layer4(self) -> None:
        """Helper for the later fine-tuning stage: unfreeze ResNet layer4 + FC."""
        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad = True

        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze the complete network for full fine-tuning."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
