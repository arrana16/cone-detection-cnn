"""AMZ-inspired three-class CNN trained from scratch on 32x32 crops."""

from __future__ import annotations

import torch
from torch import nn

from experiments.common import CLASS_NAMES, INPUT_SIZE, run_cli


MODEL_NAME = "amz_cnn"


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class AMZConeClassifier(nn.Module):
    """Three convolution blocks followed by AMZ's four dense layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, 32, 7)
        self.conv2 = ConvBlock(32, 64, 5)
        self.conv3 = ConvBlock(64, 128, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 128, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.output = nn.Linear(16, len(CLASS_NAMES))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
            raise ValueError(f"images must have shape [batch, 3, 32, 32], got {tuple(images.shape)}")
        features = self.conv1(images)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.flatten(features)
        features = self.activation(self.fc1(features))
        features = self.activation(self.fc2(features))
        features = self.activation(self.fc3(features))
        return self.output(features)


def build_model(pretrained: bool = False) -> AMZConeClassifier:
    if pretrained:
        raise ValueError("the AMZ CNN is trained from scratch")
    return AMZConeClassifier()


def main(argv: list[str] | None = None) -> int:
    return run_cli(
        argv,
        model_name=MODEL_NAME,
        build_model=build_model,
        learning_rate=0.001,
        model_settings={
            "pretrained_backbone": False,
            "architecture": [
                "conv7x7-32",
                "pool2x2",
                "conv5x5-64",
                "pool2x2",
                "conv3x3-128",
                "pool2x2",
                "flatten2048",
                "fc256",
                "fc64",
                "fc16",
                "fc3",
            ],
            "conv_dropout": 0.2,
            "flatten_features": 2048,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
