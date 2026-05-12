"""
models/model.py
───────────────
Builds the classification backbone:
  - Loads a pre-trained EfficientNet-B3 or ResNet-50
  - Partially freezes backbone layers for transfer learning
  - Replaces the final head with a task-specific classification head
"""

import logging

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """
    Load a pre-trained backbone, partially freeze it, then attach a custom
    classification head sized for ``cfg["num_classes"]``.

    Parameters
    ----------
    cfg : dict
        Must contain:
          - ``model_name``     : "efficientnet_b3" | "resnet50"
          - ``num_classes``    : int
          - ``freeze_fraction``: float  0 = train all, 1 = freeze all backbone

    Returns
    -------
    nn.Module  (weights on CPU; caller is responsible for .to(device))
    """
    num_classes    = cfg["num_classes"]
    model_name     = cfg["model_name"].lower()
    freeze_frac    = cfg["freeze_fraction"]

    # ── EfficientNet-B3 ───────────────────────────────────────────────────────
    if model_name == "efficientnet_b3":
        model      = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features   # 1536

        # Freeze the first `freeze_frac` proportion of feature blocks
        all_blocks = list(model.features.children())
        n_freeze   = int(len(all_blocks) * freeze_frac)
        for block in all_blocks[:n_freeze]:
            for param in block.parameters():
                param.requires_grad = False
        logger.info(
            "EfficientNet-B3: froze %d / %d feature blocks (freeze_fraction=%.2f).",
            n_freeze, len(all_blocks), freeze_frac,
        )

        # Custom head: Dropout → FC → SiLU → Dropout → logits
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(),           # smooth activation consistent with EfficientNet body
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    # ── ResNet-50 ─────────────────────────────────────────────────────────────
    elif model_name == "resnet50":
        model       = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features   # 2048

        # Freeze everything except layer3, layer4, and the new FC head
        for name, param in model.named_parameters():
            if not any(k in name for k in ("layer3", "layer4", "fc")):
                param.requires_grad = False
        logger.info("ResNet-50: froze conv1, bn1, layer1, layer2.")

        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    else:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'. "
            "Choose 'efficientnet_b3' or 'resnet50'."
        )

    # ── parameter summary ─────────────────────────────────────────────────────
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Parameters — Total: %s  |  Trainable: %s  (%.1f%%)",
        f"{total:,}", f"{trainable:,}", 100 * trainable / total,
    )

    return model
