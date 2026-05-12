"""
data/dataset.py
───────────────
Handles all data concerns:
  - Quality-aware preprocessing (Ben Graham + CLAHE)
  - Augmentation pipelines (train vs val)
  - ImageFolder dataset construction
  - Class-weight computation for imbalanced labels
  - DataLoader creation
"""

import logging
import os
import numpy as np
import pandas as pd
import cv2

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  QUALITY-AWARE PREPROCESSING TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

class BenGrahamPreprocess:
    """
    Ben Graham's retinal fundus preprocessing (two steps):

    1. Circular crop — zeros out the black border outside the retinal disc
       so the model never learns from camera-specific padding artefacts.

    2. Local-average subtraction — subtracts a heavily blurred version of
       the image and re-centres pixel values at 128.  This removes
       camera-specific lighting gradients and makes vessel / lesion
       contrast invariant to imaging hardware — directly addressing the
       brightness & red-channel confound identified in the EDA.

    Input / output: PIL RGB Image.
    """

    def __init__(self, target_size: int, sigma_scale: float = 10.0):
        self.target_size = target_size    # resize longest axis to this before processing
        self.sigma_scale = sigma_scale    # blur sigma = radius / sigma_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)               # H×W×3  uint8

        # ── 1. resize so the retinal circle fills the target canvas ──────────
        h, w  = arr.shape[:2]
        scale = self.target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ── 2. circular mask — zero outside the retinal disc ─────────────────
        h, w  = arr.shape[:2]
        cx, cy = w // 2, h // 2
        radius = min(cx, cy)
        y_idx, x_idx = np.ogrid[:h, :w]
        outside = ((x_idx - cx) ** 2 + (y_idx - cy) ** 2) > radius ** 2
        arr[outside] = 0

        # ── 3. local-average subtraction (Ben Graham's key step) ─────────────
        sigma     = max(radius / self.sigma_scale, 1.0)
        blurred   = cv2.GaussianBlur(arr, (0, 0), sigma)
        processed = np.clip(
            arr.astype(np.int16) * 4 - blurred.astype(np.int16) * 4 + 128,
            0, 255,
        ).astype(np.uint8)
        processed[outside] = 0           # restore circle mask after arithmetic

        return Image.fromarray(processed)


class CLAHETransform:
    """
    Contrast-Limited Adaptive Histogram Equalisation on the LAB L-channel.

    Normalises local contrast without oversaturating — breaks the
    sharpness / brightness confound (EDA finding: sharpness r=−0.517
    with DR grade, p<0.001) so the model cannot classify grade by image
    quality alone.

    Input / output: PIL RGB Image.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid  = tile_grid
        # NOTE: cv2.CLAHE is not picklable, so it cannot be stored as an
        # instance attribute when num_workers > 0 on Windows (spawn).
        # The object is created lazily per-call instead.

    def __call__(self, img: Image.Image) -> Image.Image:
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_grid)
        arr = np.array(img)                          # H×W×3 uint8 RGB
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(image_size: int, split: str, cfg: dict = None) -> transforms.Compose:
    """
    Returns a quality-aware augmentation pipeline for retinal fundus photographs.

    Preprocessing (both splits)
    ───────────────────────────
    1. BenGrahamPreprocess — circle-crop + local-average subtraction.
       Removes camera lighting gradient; breaks brightness/red-channel confound.
    2. CLAHETransform — adaptive contrast normalisation on the LAB L-channel.
       Breaks the sharpness confound (r=−0.517, p<0.001 from EDA).

    Train augmentations (on top of preprocessing)
    ──────────────────────────────────────────────
    3. Geometric  — flip, rotate, affine (retinal anatomy is rotationally invariant).
    4. ColorJitter (±0.3 brightness/contrast) — simulates camera & lighting variation.
    5. RandomApply GaussianBlur (p=0.25) — simulates focus / media opacity variation
       WITHOUT the model exploiting systematic sharpness differences by grade.

    Val — preprocessing only, then resize + normalise (deterministic).

    cfg keys used (falls back to CONFIG defaults if cfg is None)
    ────────────────────────────────────────────────────────────
    use_ben_graham    : bool  (default True)
    use_clahe         : bool  (default True)
    clahe_clip_limit  : float (default 2.0)
    clahe_tile_grid   : tuple (default (8, 8))
    """
    from config import CONFIG
    _cfg = cfg if cfg is not None else CONFIG

    use_ben_graham   = _cfg.get("use_ben_graham",   True)
    use_clahe        = _cfg.get("use_clahe",        True)
    clahe_clip_limit = _cfg.get("clahe_clip_limit", 2.0)
    clahe_tile_grid  = _cfg.get("clahe_tile_grid",  (8, 8))

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # ── shared quality-aware preprocessing (applied to every split) ──────────
    preprocess_steps = []
    if use_ben_graham:
        preprocess_steps.append(
            BenGrahamPreprocess(target_size=image_size, sigma_scale=10.0)
        )
    if use_clahe:
        preprocess_steps.append(
            CLAHETransform(clip_limit=clahe_clip_limit, tile_grid=clahe_tile_grid)
        )

    if split == "train":
        return transforms.Compose([
            *preprocess_steps,

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

            # Colour — ±0.3 brightness/contrast to simulate camera variation
            # (strengthened from ±0.2 — EDA showed systematic R/G intensity
            #  differences across grades that must not become a learned shortcut)
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.15,
                hue=0.03,
            ),

            # Blur — RandomApply (p=0.25) so the model sees both sharp and
            # soft images for every grade, preventing sharpness-based shortcuts
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))],
                p=0.25,
            ),

            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),

            # Random erasing — simulates optic-disc / vessel occlusion artefacts
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.05)),
        ])
    else:   # val / test — deterministic, NO heavy preprocessing
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET & DATALOADER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class RetinalDataset(Dataset):
    """
    Custom Dataset for the APTOS / flat-folder format:

        data/train.csv          ← columns: id_code, diagnosis  (int 0-4)
        data/train_images/      ← <id_code>.png  (all images in one directory)

    This replaces torchvision.datasets.ImageFolder which requires images to be
    pre-sorted into class-labelled subdirectories — a layout this dataset does
    NOT have.

    Parameters
    ──────────
    df          : DataFrame with columns ['id_code', 'diagnosis']
    images_dir  : directory containing <id_code>.png files
    transform   : torchvision transform pipeline
    """

    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["id_code"] + ".png")
        img      = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["diagnosis"])

    @property
    def targets(self) -> list[int]:
        """List of integer labels — same interface as ImageFolder.targets."""
        return self.df["diagnosis"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET & DATALOADER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train / val / test DataLoaders from a single labelled CSV.

    Why three splits from train.csv?
    ──────────────────────────────────
    The Kaggle test_images have no labels — they can only produce a submission
    CSV, not real metric estimates.  Holding out a stratified test split from
    the labelled data gives a *truly untouched* set to measure generalisation.

      train (80%) — model sees these images during forward/backward pass
      val   (10%) — used for LR scheduling & early stopping (indirectly touched)
      test  (10%) — never seen during training; final unbiased evaluation only

    An 80/10/10 split is used rather than 70/15/15 to preserve more training
    data.  With only ~194 Grade-3 images (5.3% of 3,662), a 15% test slice
    would leave ~29 Grade-3 test samples — too few for stable metric estimates.

    Both splits use StratifiedShuffleSplit to keep class proportions identical
    across all three subsets.

    Returns
    ───────
    train_loader  : DataLoader  (WeightedRandomSampler if use_weighted_sampler)
    val_loader    : DataLoader  (deterministic)
    test_loader   : DataLoader  (deterministic, held-out)
    class_weights : FloatTensor (num_classes,) — inverse-frequency weights for loss
    """
    df = pd.read_csv(cfg["train_csv"])
    images_dir = cfg["train_images_dir"]

    val_split  = cfg.get("val_split",  0.1)
    test_split = cfg.get("test_split", 0.1)

    # ── stage 1: carve out the held-out test set ──────────────────────────────
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_split, random_state=cfg["seed"]
    )
    trainval_idx, test_idx = next(sss1.split(df, df["diagnosis"]))
    trainval_df = df.iloc[trainval_idx]
    test_df     = df.iloc[test_idx]

    # ── stage 2: split remaining data into train / val ────────────────────────
    # val_split is expressed as a fraction of the *original* dataset, so we
    # rescale it to be a fraction of the train+val pool.
    val_fraction = val_split / (1.0 - test_split)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=cfg["seed"]
    )
    train_idx, val_idx = next(sss2.split(trainval_df, trainval_df["diagnosis"]))
    train_df = trainval_df.iloc[train_idx]
    val_df   = trainval_df.iloc[val_idx]

    train_dataset = RetinalDataset(
        train_df, images_dir,
        transform=get_transforms(cfg["image_size"], "train", cfg),
    )
    val_dataset = RetinalDataset(
        val_df, images_dir,
        transform=get_transforms(cfg["image_size"], "val", cfg),
    )
    test_dataset = RetinalDataset(
        test_df, images_dir,
        transform=get_transforms(cfg["image_size"], "val", cfg),  # deterministic
    )

    # ── inverse-frequency class weights (for CrossEntropyLoss) ───────────────
    class_counts         = np.bincount(train_dataset.targets, minlength=cfg["num_classes"])
    val_counts           = np.bincount(val_dataset.targets,   minlength=cfg["num_classes"])
    test_counts          = np.bincount(test_dataset.targets,  minlength=cfg["num_classes"])
    class_weights        = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()   # add this line
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # ── class distribution table — printed before training starts ─────────────
    logger.info("=" * 72)
    logger.info("SPLIT CLASS DISTRIBUTION (stratified 80 / 10 / 10)")
    logger.info("%-24s  %8s  %8s  %8s", "Class", "Train", "Val", "Test")
    logger.info("-" * 52)
    for i, name in enumerate(cfg["class_names"]):
        logger.info(
            "%-24s  %8d  %8d  %8d",
            name, class_counts[i], val_counts[i], test_counts[i],
        )
    logger.info("-" * 52)
    logger.info(
        "%-24s  %8d  %8d  %8d",
        "TOTAL", class_counts.sum(), val_counts.sum(), test_counts.sum(),
    )
    logger.info("=" * 72)

    # ── WeightedRandomSampler — fixes imbalance at the batch level ───────────
    use_weighted_sampler = cfg.get("use_weighted_sampler", True)
    if use_weighted_sampler:
        sample_weights = torch.tensor(
            [class_weights[t] for t in train_dataset.targets], dtype=torch.float32
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        logger.info("WeightedRandomSampler enabled (replacement=True).")
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights_tensor


