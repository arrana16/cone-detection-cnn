"""ImageNet transfer learning with EfficientNet V2 S on 32x32 cone crops."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from experiments.common import CLASS_NAMES, INPUT_SIZE, run_cli


MODEL_NAME = "efficientnet_v2_s"


class EfficientNetV2SConeClassifier(nn.Module):
    """Pretrained EfficientNet V2 S with a three-class cone output layer."""

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.network = models.efficientnet_v2_s(weights=weights)
        in_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(in_features, len(CLASS_NAMES))
        for parameter in self.network.parameters():
            parameter.requires_grad = False
        for parameter in self.network.classifier[-1].parameters():
            parameter.requires_grad = True

    def set_frozen_backbone_eval(self) -> None:
        """Keep frozen feature batch-normalization statistics unchanged."""

        self.network.features.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
            raise ValueError(f"images must have shape [batch, 3, 32, 32], got {tuple(images.shape)}")
        return self.network(images)


def build_model(pretrained: bool = True) -> EfficientNetV2SConeClassifier:
    return EfficientNetV2SConeClassifier(pretrained=pretrained)


def main(argv: list[str] | None = None) -> int:
    return run_cli(
        argv,
        model_name=MODEL_NAME,
        build_model=build_model,
        learning_rate=0.003,
        model_settings={
            "pretrained_backbone": True,
            "trainable_parameters": "final classifier layer only",
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
