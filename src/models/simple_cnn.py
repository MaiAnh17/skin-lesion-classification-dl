import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for HAM10000 skin lesion classification.

    Architecture:
        Conv Block 1:
            Conv2d(3 -> 32)
            ReLU
            MaxPool

        Conv Block 2:
            Conv2d(32 -> 64)
            ReLU
            MaxPool

        Conv Block 3:
            Conv2d(64 -> 128)
            ReLU
            MaxPool

        Adaptive Average Pooling:
            128 x 28 x 28
            -> 128 x 4 x 4

        Classifier:
            Flatten
            Linear(2048 -> 256)
            ReLU
            Dropout
            Linear(256 -> num_classes)
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout: float = 0.5,
    ):
        super().__init__()

        # =====================================================
        # Convolution Block 1
        # =====================================================

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # =====================================================
        # Convolution Block 2
        # =====================================================

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # =====================================================
        # Convolution Block 3
        # =====================================================

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu3 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # =====================================================
        # Adaptive pooling
        # =====================================================

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            output_size=(4, 4)
        )

        # =====================================================
        # Fully Connected Classifier
        # =====================================================

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            128 * 4 * 4,
            256,
        )

        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.fc2 = nn.Linear(
            256,
            num_classes,
        )

    def forward(self, x):
        # -----------------------------------------
        # Block 1
        # -----------------------------------------

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # -----------------------------------------
        # Block 2
        # -----------------------------------------

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # -----------------------------------------
        # Block 3
        # -----------------------------------------

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # -----------------------------------------
        # Adaptive pooling
        # -----------------------------------------

        x = self.adaptive_pool(x)

        # -----------------------------------------
        # Fully connected classifier
        # -----------------------------------------

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x