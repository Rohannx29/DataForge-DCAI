"""
Model architecture definitions.

Uses `timm` pretrained backbones. Kept as a single factory function so the
exact same architecture instantiation is guaranteed reproducible across every
experimental condition — no per-experiment architecture tweaks are permitted.
"""
import timm
import torch.nn as nn


def build_model(
    architecture: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    """Instantiate a classification model from a timm backbone.

    Args:
        architecture: timm model name, e.g. "resnet18", "efficientnet_b0".
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet-pretrained weights.
        freeze_backbone: If True, freezes all layers except the final classifier head.
        dropout: Dropout probability applied before the final classification layer.

    Returns:
        A timm model instance configured for num_classes-way classification.
    """
    model = timm.create_model(
        architecture,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name and "head" not in name:
                param.requires_grad = False

    return model
