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
        hidden_dim: int | None = 256,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features  # 512

        if hidden_dim is not None and hidden_dim > 0:
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load state dict, dynamically adapting the FC head structure if necessary."""
        has_2layer = any("fc.3." in k for k in state_dict.keys())
        has_1layer = any("fc.1." in k for k in state_dict.keys())
        in_features = self.backbone.fc[0].in_features if isinstance(self.backbone.fc[0], nn.Linear) else 512

        if has_2layer and not isinstance(self.backbone.fc[0], nn.Linear):
            out_dim = state_dict[[k for k in state_dict if "fc.0.weight" in k][0]].shape[0]
            num_classes = state_dict[[k for k in state_dict if "fc.3.weight" in k][0]].shape[0]
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(out_dim, num_classes),
            )
        elif has_1layer and isinstance(self.backbone.fc[0], nn.Linear):
            num_classes = state_dict[[k for k in state_dict if "fc.1.weight" in k][0]].shape[0]
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )

        return super().load_state_dict(state_dict, strict=strict, assign=assign)
