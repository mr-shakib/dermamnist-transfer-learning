"""Model zoo: a from-scratch CNN baseline and ImageNet-pretrained backbones."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from common import NUM_CLASSES


class SimpleCNN(nn.Module):
    """Five conv blocks (conv-BN-ReLU x2 + max-pool) followed by global average pooling."""

    def __init__(self, num_classes: int = NUM_CLASSES, widths=(16, 32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 3
        for c_out in widths:
            layers += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


def build_model(arch: str, pretrained: bool, dropout: float = 0.3) -> nn.Module:
    if arch == "simplecnn":
        return SimpleCNN(dropout=dropout)
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, NUM_CLASSES))
        return m
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, NUM_CLASSES))
        return m
    raise ValueError(f"unknown arch {arch}")


def head_module(model: nn.Module, arch: str) -> nn.Module:
    return {"simplecnn": lambda: model.head, "resnet50": lambda: model.fc,
            "efficientnet_b0": lambda: model.classifier}[arch]()


def backbone_stages(model: nn.Module, arch: str) -> list[nn.Module]:
    """Backbone split into stages ordered from input to output (used for gradual unfreezing)."""
    if arch == "resnet50":
        stem = nn.ModuleList([model.conv1, model.bn1, model.layer1])
        return [stem, model.layer2, model.layer3, model.layer4]
    if arch == "efficientnet_b0":
        f = model.features  # 9 blocks: stem, 7 MBConv stages, head conv
        return [nn.ModuleList(list(f[:3])), nn.ModuleList(list(f[3:5])), nn.ModuleList(list(f[5:7])),
                nn.ModuleList(list(f[7:]))]
    if arch == "simplecnn":
        return [model.features]
    raise ValueError(arch)


def embed(model: nn.Module, arch: str, x: torch.Tensor) -> torch.Tensor:
    """Penultimate-layer embedding (input to the classification head)."""
    if arch == "simplecnn":
        return model.forward_features(x)
    if arch == "resnet50":
        m = model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return torch.flatten(m.avgpool(x), 1)
    if arch == "efficientnet_b0":
        return torch.flatten(model.avgpool(model.features(x)), 1)
    raise ValueError(arch)


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    return {"resnet50": lambda: model.layer4[-1], "efficientnet_b0": lambda: model.features[-1],
            "simplecnn": lambda: model.features[-3]}[arch]()


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
