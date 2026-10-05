import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """
    A reusable CNN block:
    Conv2d -> BatchNorm -> ReLU
    -> Conv2d -> BatchNorm -> ReLU
    -> MaxPool -> Dropout2d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Dropout2d(
                p=dropout
            ),
        )

    def forward(self, x):

        x = self.block(x)

        return x


class ComplexCNN(nn.Module):
    """
    Complex CNN for HAM10000 classification.

    Input:
        [batch_size, 3, 224, 224]

    Output:
        [batch_size, 7]
    """

    def __init__(
        self,
        num_classes=7,
        dropout=0.5,
    ):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(

            # Input: 3 x 224 x 224
            # Output: 32 x 112 x 112
            CNNBlock(
                in_channels=3,
                out_channels=32,
                dropout=0.10,
            ),

            # Output: 64 x 56 x 56
            CNNBlock(
                in_channels=32,
                out_channels=64,
                dropout=0.15,
            ),

            # Output: 128 x 28 x 28
            CNNBlock(
                in_channels=64,
                out_channels=128,
                dropout=0.20,
            ),

            # Output: 256 x 14 x 14
            CNNBlock(
                in_channels=128,
                out_channels=256,
                dropout=0.25,
            ),
        )

        # Reduce feature map:
        # 256 x 14 x 14
        # ->
        # 256 x 2 x 2
        self.avgpool = nn.AdaptiveAvgPool2d(
            (2, 2)
        )

        # Classification head
        self.classifier = nn.Sequential(

            nn.Flatten(),

            # 256 * 2 * 2 = 1024
            nn.Linear(
                256 * 2 * 2,
                256,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                256,
                num_classes,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = self.classifier(x)

        return x