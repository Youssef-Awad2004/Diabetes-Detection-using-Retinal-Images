"""
data/dataset.py
───────────────
Handles all data concerns:
  - Augmentation pipelines (train vs val)
  - ImageFolder dataset construction
  - Class-weight computation for imbalanced labels
  - DataLoader creation
"""

import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(image_size: int, split: str) -> transforms.Compose:
    """
    Returns an augmentation pipeline tailored for retinal fundus photographs.

    Train  — geometric + colour augmentations to improve robustness.
    Val    — only resize + normalise (no stochastic ops).

    ImageNet mean/std are used because the backbone was pre-trained on it.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            # Resize slightly larger then random-crop → natural scale jitter
            transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
            transforms.RandomCrop(image_size),

            # Geometric — retinal anatomy is rotationally invariant
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),

            # Colour — simulate different fundus cameras / lighting
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02,
            ),

            # Gaussian blur — mimics focus / media opacity variation
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),

            # Random erasing — simulates optic-disc / vessel occlusion artefacts
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.05)),
        ])

    else:   # val / test — deterministic
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET & DATALOADER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Build train and validation DataLoaders from an ImageFolder directory tree.

    Expected directory structure
    ────────────────────────────
    data/
      train/
        Normal/           img1.jpg  img2.jpg …
        Mild DR/          …
        Moderate DR/      …
        Severe DR/        …
        Proliferative DR/ …
      val/
        Normal/           …
        …

    Returns
    -------
    train_loader  : DataLoader
    val_loader    : DataLoader
    class_weights : 1-D FloatTensor of shape (num_classes,)
                    Inverse-frequency weights; pass to CrossEntropyLoss.
    """
    train_dataset = datasets.ImageFolder(
        root=cfg["train_dir"],
        transform=get_transforms(cfg["image_size"], "train"),
    )
    val_dataset = datasets.ImageFolder(
        root=cfg["val_dir"],
        transform=get_transforms(cfg["image_size"], "val"),
    )

    # ── inverse-frequency class weights (handles label imbalance) ─────────────
    class_counts  = np.bincount(train_dataset.targets)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    logger.info(
        "Class distribution (train): %s",
        dict(zip(cfg["class_names"], class_counts.tolist())),
    )
    logger.info(
        "Train samples: %d  |  Val samples: %d",
        len(train_dataset), len(val_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,     # avoids single-sample batches that break BatchNorm
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, class_weights
