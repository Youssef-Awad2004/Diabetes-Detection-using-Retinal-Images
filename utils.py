"""
utils/utils.py
──────────────
Shared utilities: reproducibility seeding, device selection, and MixUp.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Make the entire run deterministic (CPU + CUDA)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    logger.info("Global seed set to %d", seed)


def get_device() -> torch.device:
    """
    Return the best available device in priority order:
    CUDA GPU → Apple MPS → CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Device: CUDA — %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU")
    return device


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp data augmentation for ordinal DR grading.

    Why MixUp for DR specifically
    ──────────────────────────────
    DR grades are clinically ordinal: Grade 1 and Grade 2 share many visual
    features.  MixUp linearly interpolates both images and labels between two
    samples, creating soft "in-between" training examples (e.g. 60% Grade-1 +
    40% Grade-2).  This teaches the model the severity continuum rather than
    treating each grade as a hard-boundary category — directly addressing
    EDA Finding #2 (ordinal label structure).

    Usage in the training loop
    ──────────────────────────
        images, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=cfg["mixup_alpha"])
        logits = model(images)
        loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

    Parameters
    ──────────
    images : FloatTensor  (B, C, H, W)
    labels : LongTensor   (B,)
    alpha  : Beta distribution parameter.  0 = no mixing; 0.4 is a good default
             for medical imaging (aggressive enough to regularise, mild enough
             not to destroy lesion signals).

    Returns
    ───────
    mixed_images : FloatTensor (B, C, H, W)
    labels_a     : LongTensor  (B,)   — original labels
    labels_b     : LongTensor  (B,)   — shuffled labels
    lam          : float               — mixing coefficient
    """
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam
