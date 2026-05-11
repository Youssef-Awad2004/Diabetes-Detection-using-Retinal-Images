"""
utils/utils.py
──────────────
Shared utilities: reproducibility seeding and device selection.
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
