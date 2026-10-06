"""PyTorch contract and image preprocessing for cone-colour crops.

The model consumes normalized RGB tensors shaped ``[N, 3, 96, 96]`` and
returns raw logits ordered as ``blue``, ``yellow``, ``other``.  Boxes can be
turned into square crops with 15% context using :func:`crop_from_box`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models


CLASS_NAMES = ("blue", "yellow", "other")
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE = 96
CONTEXT_FRACTION = 0.15
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_PAD_RGB = tuple(round(channel * 255) for channel in IMAGENET_MEAN)


class Prediction(NamedTuple):
    """The winning class and its probabilities in ``CLASS_NAMES`` order."""

    label: str
    probabilities: dict[str, float]


class ConeColorClassifier(nn.Module):
    """EfficientNet B0 with a three-class cone-colour head."""

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.network = models.efficientnet_b0(weights=weights)
        in_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for normalized ``[N, 3, 96, 96]`` images."""

        expected_tail = (3, INPUT_SIZE, INPUT_SIZE)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_tail:
            raise ValueError(
                "images must have shape [batch, 3, 96, 96]; "
                f"received {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise TypeError("images must be a floating-point tensor")
        return self.network(images)


def build_model(*, pretrained: bool = True) -> ConeColorClassifier:
    """Create the classifier; pretrained weights download if not cached."""

    return ConeColorClassifier(pretrained=pretrained)


def crop_from_box(
    image: Image.Image,
    box_xyxy: Sequence[float],
    *,
    context_fraction: float = CONTEXT_FRACTION,
) -> Image.Image:
    """Extract a square RGB crop with context around an ``(x1, y1, x2, y2)`` box.

    The context fraction is added as a margin on each side of the longer box
    dimension. Pixels beyond the source image are padded with the RGB value
    corresponding approximately to zero after ImageNet normalization.
    """

    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("box coordinates must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    if not math.isfinite(context_fraction) or context_fraction < 0:
        raise ValueError("context_fraction must be finite and non-negative")

    side = max(1, math.ceil(max(x2 - x1, y2 - y1) * (1 + 2 * context_fraction)))
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = math.floor(center_x - side / 2)
    top = math.floor(center_y - side / 2)
    source = image.convert("RGB")
    square = Image.new("RGB", (side, side), _PAD_RGB)

    source_left, source_top = max(0, left), max(0, top)
    source_right = min(source.width, left + side)
    source_bottom = min(source.height, top + side)
    if source_left < source_right and source_top < source_bottom:
        patch = source.crop((source_left, source_top, source_right, source_bottom))
        square.paste(patch, (source_left - left, source_top - top))
    return square


def preprocess_crop(crop: Image.Image) -> torch.Tensor:
    """Convert a contextual RGB crop to normalized ``[3, 96, 96]`` float data."""

    rgb = crop.convert("RGB")
    side = max(rgb.size)
    square = Image.new("RGB", (side, side), _PAD_RGB)
    offset = ((side - rgb.width) // 2, (side - rgb.height) // 2)
    square.paste(rgb, offset)
    resized = square.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)

    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def predict_crop(model: ConeColorClassifier, crop: Image.Image) -> Prediction:
    """Classify one RGB crop that already contains the desired context."""

    model.eval()
    device = next(model.parameters()).device
    tensor = preprocess_crop(crop).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu()
    probability_map = {
        name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)
    }
    label = CLASS_NAMES[int(probabilities.argmax())]
    return Prediction(label=label, probabilities=probability_map)


def predict_box(
    model: ConeColorClassifier,
    image: Image.Image,
    box_xyxy: Sequence[float],
) -> Prediction:
    """Extract a box crop with the standard context and return its prediction."""

    return predict_crop(model, crop_from_box(image, box_xyxy))
