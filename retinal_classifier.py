"""
=============================================================================
  Diabetic Retinopathy Grading — EfficientNet-B3 Fine-Tuning Pipeline
=============================================================================
  Task   : 5-class classification (0=Normal, 1=Mild, 2=Moderate,
           3=Severe, 4=Proliferative DR)
  Model  : EfficientNet-B3 (ImageNet pre-trained, torchvision)
  Author : Generated for clinical-grade DR grading
  Usage  : python retinal_classifier.py
           Edit the CONFIG section below before running.
=============================================================================
"""

import os
import copy
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — works on servers without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B3_Weights

from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  — edit this section before running
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Paths  ──────────────────────────────────────────────────────────────────
    "train_dir"      : "data/train",        # ImageFolder: train/class_name/img.jpg
    "val_dir"        : "data/val",          # ImageFolder: val/class_name/img.jpg
    "checkpoint_dir" : "checkpoints",       # where .pth files are saved
    "results_dir"    : "results",           # where plots / reports are saved

    # Model  ──────────────────────────────────────────────────────────────────
    "num_classes"    : 5,
    "model_name"     : "efficientnet_b3",   # or "resnet50"

    # Training  ───────────────────────────────────────────────────────────────
    "epochs"         : 40,
    "batch_size"     : 32,
    "num_workers"    : 4,
    "image_size"     : 300,                 # EfficientNet-B3 native resolution

    # Optimiser  ──────────────────────────────────────────────────────────────
    "lr"             : 3e-4,
    "weight_decay"   : 1e-4,
    "lr_patience"    : 5,                   # ReduceLROnPlateau patience
    "lr_factor"      : 0.3,
    "min_lr"         : 1e-7,

    # Transfer learning  ──────────────────────────────────────────────────────
    # Fraction of backbone blocks to FREEZE (0 = train all, 1 = freeze all)
    "freeze_fraction": 0.7,

    # Misc  ───────────────────────────────────────────────────────────────────
    "seed"           : 42,
    "label_smoothing": 0.1,                 # helps with inter-grader variability
    "class_names"    : [
        "Normal", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"
    ],
}
# ─────────────────────────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    """Make training deterministic."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
#  1.  DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(image_size: int, split: str) -> transforms.Compose:
    """
    Returns augmentation pipeline tailored for retinal fundus photographs.

    Train  — geometric + colour augmentations to improve robustness.
    Val    — only resize / crop + normalise (no stochastic ops).

    ImageNet mean/std are used because the backbone was pre-trained on it;
    fine-tuning benefits from keeping the same normalisation statistics.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            # Resize slightly larger then random crop → natural scale jitter
            transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
            transforms.RandomCrop(image_size),

            # Geometric augmentations — retinal anatomy is rotationally invariant
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),

            # Colour augmentations — simulate camera / lighting variation
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02,
            ),

            # Occasional Gaussian blur — mimics focus variation
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),

            # Random erasing simulates optic-disc/vessel occlusion artefacts
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.05)),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and validation DataLoaders from an ImageFolder directory tree.

    Expected structure:
        data/
          train/
            Normal/        img1.jpg  img2.jpg …
            Mild DR/       …
            Moderate DR/   …
            Severe DR/     …
            Proliferative DR/ …
          val/
            Normal/        …
            …
    """
    train_dataset = datasets.ImageFolder(
        root=cfg["train_dir"],
        transform=get_transforms(cfg["image_size"], "train"),
    )
    val_dataset = datasets.ImageFolder(
        root=cfg["val_dir"],
        transform=get_transforms(cfg["image_size"], "val"),
    )

    # ── compute class weights to handle label imbalance ───────────────────────
    class_counts  = np.bincount(train_dataset.targets)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    logger.info("Class distribution (train): %s", dict(zip(cfg["class_names"], class_counts.tolist())))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,          # avoid single-sample last batch with BN
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    logger.info("Train samples: %d  |  Val samples: %d",
                len(train_dataset), len(val_dataset))

    return train_loader, val_loader, class_weights


# ─────────────────────────────────────────────────────────────────────────────
#  2.  MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """
    Load a pre-trained backbone, partially freeze it, and attach a custom
    classification head for the 5-class DR grading task.
    """
    num_classes = cfg["num_classes"]
    model_name  = cfg["model_name"].lower()

    # ── load backbone ─────────────────────────────────────────────────────────
    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features   # 1536

        # ── freeze early feature-extractor blocks ─────────────────────────────
        #    EfficientNet features = model.features (list of blocks 0-8)
        all_blocks  = list(model.features.children())
        n_freeze    = int(len(all_blocks) * cfg["freeze_fraction"])
        for block in all_blocks[:n_freeze]:
            for param in block.parameters():
                param.requires_grad = False
        logger.info("EfficientNet-B3: froze %d / %d feature blocks.", n_freeze, len(all_blocks))

        # ── replace classification head ───────────────────────────────────────
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.SiLU(),                    # smooth activation — matches EfficientNet body
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),  # raw logits; softmax applied at inference
        )

    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features   # 2048

        # ── freeze all layers except layer3, layer4, fc ───────────────────────
        for name, param in model.named_parameters():
            if not any(k in name for k in ("layer3", "layer4", "fc")):
                param.requires_grad = False
        logger.info("ResNet-50: froze layers conv1, bn1, layer1, layer2.")

        # ── replace FC head ───────────────────────────────────────────────────
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"Unsupported model_name: '{model_name}'. "
                         "Choose 'efficientnet_b3' or 'resnet50'.")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total params: %s  |  Trainable: %s  (%.1f%%)",
                f"{total_params:,}", f"{trainable_params:,}",
                100 * trainable_params / total_params)

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  3.  LOSS & OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: dict,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple:
    """
    Returns:
        criterion  : CrossEntropyLoss with label smoothing + class weights
        optimizer  : AdamW with differential learning rates
                     (backbone uses 0.1× of head LR — common fine-tune trick)
        scheduler  : ReduceLROnPlateau watching val-loss
    """
    # ── loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=cfg["label_smoothing"],
    )

    # ── param groups: lower LR for frozen/backbone layers ─────────────────────
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["lr"] * 0.1},
            {"params": head_params,     "lr": cfg["lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["lr_factor"],
        patience=cfg["lr_patience"],
        min_lr=cfg["min_lr"],
        verbose=True,
    )

    return criterion, optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
#  4.  TRAINING & VALIDATION LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    phase: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    One epoch of training or validation.

    Returns
    -------
    avg_loss   : mean cross-entropy over all batches
    all_labels : ground-truth integer labels  (N,)
    all_preds  : argmax predictions           (N,)
    all_probs  : softmax probabilities        (N, C)
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── forward ───────────────────────────────────────────────────────
            logits = model(images)                    # (B, num_classes)
            loss   = criterion(logits, labels)

            # ── backward (train only) ─────────────────────────────────────────
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # ── collect stats ─────────────────────────────────────────────────
            running_loss += loss.item() * images.size(0)
            probs  = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds  = np.argmax(probs, axis=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)

    avg_loss   = running_loss / len(loader.dataset)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.vstack(all_probs)

    return avg_loss, all_labels, all_preds, all_probs


# ─────────────────────────────────────────────────────────────────────────────
#  5.  CLINICAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    preds:  np.ndarray,
    probs:  np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Compute accuracy, macro AUC-ROC, per-class sensitivity & specificity,
    and the full confusion matrix.

    Sensitivity (Recall) and Specificity are derived from the confusion matrix
    in a one-vs-rest manner — standard practice in clinical reporting.
    """
    num_classes   = len(class_names)
    accuracy      = accuracy_score(labels, preds)

    # ── AUC-ROC (macro, one-vs-rest) ─────────────────────────────────────────
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    try:
        auc_macro = roc_auc_score(labels_bin, probs,
                                  multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = float("nan")   # can happen if a class is missing in val batch

    # ── confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # ── per-class sensitivity & specificity from CM ───────────────────────────
    sensitivity_per_class = []
    specificity_per_class = []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        sens = TP / (TP + FN + 1e-8)
        spec = TN / (TN + FP + 1e-8)
        sensitivity_per_class.append(sens)
        specificity_per_class.append(spec)

    macro_sensitivity = np.mean(sensitivity_per_class)
    macro_specificity = np.mean(specificity_per_class)

    return {
        "accuracy"              : accuracy,
        "auc_roc"               : auc_macro,
        "macro_sensitivity"     : macro_sensitivity,
        "macro_specificity"     : macro_specificity,
        "sensitivity_per_class" : sensitivity_per_class,
        "specificity_per_class" : specificity_per_class,
        "confusion_matrix"      : cm,
        "labels_bin"            : labels_bin,
        "probs"                 : probs,
    }


def log_metrics(phase: str, epoch: int, loss: float, metrics: dict,
                class_names: list[str]) -> None:
    """Pretty-print a one-line summary + per-class sensitivity/specificity."""
    logger.info(
        "[%s] Epoch %3d | Loss: %.4f | Acc: %.4f | AUC: %.4f | "
        "Sens: %.4f | Spec: %.4f",
        phase.upper(), epoch, loss,
        metrics["accuracy"], metrics["auc_roc"],
        metrics["macro_sensitivity"], metrics["macro_specificity"],
    )
    for i, cls in enumerate(class_names):
        logger.info(
            "         %-20s  Sensitivity: %.3f   Specificity: %.3f",
            cls,
            metrics["sensitivity_per_class"][i],
            metrics["specificity_per_class"][i],
        )


# ─────────────────────────────────────────────────────────────────────────────
#  6.  CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    cfg: dict,
    checkpoint_dir: str,
) -> None:
    """
    Save model weights + training state when validation loss improves.
    Keeps the best checkpoint and the most recent one.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_loss"   : val_loss,
        "config"     : cfg,
    }
    path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch:03d}.pth")
    torch.save(checkpoint, path)
    logger.info("✓ Checkpoint saved → %s  (val_loss=%.5f)", path, val_loss)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Accumulates per-epoch metrics for plotting."""
    def __init__(self):
        self.train_loss, self.val_loss       = [], []
        self.train_acc,  self.val_acc        = [], []
        self.val_auc                         = []
        self.val_sensitivity, self.val_specificity = [], []

    def update(self, t_loss, t_acc, v_loss, v_metrics):
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.train_acc.append(t_acc)
        self.val_acc.append(v_metrics["accuracy"])
        self.val_auc.append(v_metrics["auc_roc"])
        self.val_sensitivity.append(v_metrics["macro_sensitivity"])
        self.val_specificity.append(v_metrics["macro_specificity"])


def plot_training_curves(tracker: MetricsTracker, results_dir: str) -> None:
    """Save a 2×2 grid of training/validation curves."""
    epochs = range(1, len(tracker.train_loss) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves — Diabetic Retinopathy Grading", fontsize=14, fontweight="bold")

    plot_configs = [
        (axes[0, 0], "Loss",        tracker.train_loss, tracker.val_loss,     "Train Loss",    "Val Loss",    "blue",   "red"),
        (axes[0, 1], "Accuracy",    tracker.train_acc,  tracker.val_acc,      "Train Acc",     "Val Acc",     "green",  "orange"),
        (axes[1, 0], "AUC-ROC",     None,               tracker.val_auc,      None,            "Val AUC-ROC", None,     "purple"),
        (axes[1, 1], "Sens / Spec", None,               tracker.val_sensitivity, None,         "Sensitivity", None,     "teal"),
    ]

    for ax, title, train_data, val_data, train_label, val_label, tc, vc in plot_configs:
        if train_data is not None:
            ax.plot(epochs, train_data, color=tc, label=train_label, linewidth=2)
        ax.plot(epochs, val_data, color=vc, label=val_label, linewidth=2, linestyle="--")
        if title == "Sens / Spec":
            ax.plot(epochs, tracker.val_specificity, color="coral",
                    label="Specificity", linewidth=2, linestyle=":")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Training curves saved → %s", out)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          epoch: int, results_dir: str) -> None:
    """Normalised + raw confusion matrix side-by-side."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix — Epoch {epoch}", fontweight="bold")

    for ax, data, fmt, title in [
        (ax1, cm,      "d",    "Raw Counts"),
        (ax2, cm_norm, ".2f",  "Normalised"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual",    fontweight="bold")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = os.path.join(results_dir, f"confusion_matrix_epoch{epoch:03d}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved → %s", out)


def plot_roc_curves(labels_bin: np.ndarray, probs: np.ndarray,
                    class_names: list[str], epoch: int,
                    results_dir: str) -> None:
    """One-vs-rest ROC curve per class + macro average."""
    num_classes = len(class_names)
    palette     = plt.cm.get_cmap("tab10", num_classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    mean_fpr = np.linspace(0, 1, 200)
    tprs     = []

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=palette(i), lw=1.8,
                label=f"{class_names[i]}  (AUC = {roc_auc_val:.3f})")
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    macro_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color="black", lw=2.5, linestyle="--",
            label=f"Macro Average  (AUC = {macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curves (OvR) — Epoch {epoch}", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(results_dir, f"roc_curves_epoch{epoch:03d}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("ROC curves saved → %s", out)


def plot_per_class_metrics(metrics: dict, class_names: list[str],
                            epoch: int, results_dir: str) -> None:
    """Bar chart of per-class sensitivity and specificity."""
    x     = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, metrics["sensitivity_per_class"],
           width, label="Sensitivity", color="steelblue")
    ax.bar(x + width / 2, metrics["specificity_per_class"],
           width, label="Specificity", color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-Class Sensitivity & Specificity — Epoch {epoch}",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (s, sp) in enumerate(zip(metrics["sensitivity_per_class"],
                                     metrics["specificity_per_class"])):
        ax.text(i - width / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, sp + 0.01, f"{sp:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = os.path.join(results_dir, f"per_class_metrics_epoch{epoch:03d}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Per-class metrics saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    seed_everything(cfg["seed"])

    # ── directories ───────────────────────────────────────────────────────────
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, class_weights = build_dataloaders(cfg)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # ── loss / optimiser ──────────────────────────────────────────────────────
    criterion, optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, class_weights, device
    )

    # ── training state ────────────────────────────────────────────────────────
    tracker       = MetricsTracker()
    best_val_loss = float("inf")
    best_weights  = copy.deepcopy(model.state_dict())

    logger.info("=" * 70)
    logger.info("Starting training for %d epochs", cfg["epochs"])
    logger.info("=" * 70)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # ── train ─────────────────────────────────────────────────────────────
        t_loss, t_labels, t_preds, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, "train"
        )
        t_acc = accuracy_score(t_labels, t_preds)

        # ── validate ──────────────────────────────────────────────────────────
        v_loss, v_labels, v_preds, v_probs = run_epoch(
            model, val_loader, criterion, None, device, "val"
        )
        v_metrics = compute_metrics(v_labels, v_preds, v_probs, cfg["class_names"])

        # ── LR schedule step ──────────────────────────────────────────────────
        scheduler.step(v_loss)

        # ── logging ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        logger.info("── Epoch %d/%d  (%.1fs) ──────────────────────────────",
                    epoch, cfg["epochs"], elapsed)
        log_metrics("train", epoch, t_loss, {"accuracy": t_acc, "auc_roc": float("nan"),
                                              "macro_sensitivity": float("nan"),
                                              "macro_specificity": float("nan")},
                    cfg["class_names"])
        log_metrics("val",   epoch, v_loss, v_metrics, cfg["class_names"])
        print(classification_report(
            v_labels, v_preds, target_names=cfg["class_names"], digits=4
        ))

        # ── checkpointing ─────────────────────────────────────────────────────
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights  = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, epoch, v_loss,
                            cfg, cfg["checkpoint_dir"])

        # ── metrics tracker ───────────────────────────────────────────────────
        tracker.update(t_loss, t_acc, v_loss, v_metrics)

        # ── per-epoch plots ───────────────────────────────────────────────────
        plot_confusion_matrix(v_metrics["confusion_matrix"],
                              cfg["class_names"], epoch, cfg["results_dir"])
        plot_roc_curves(v_metrics["labels_bin"], v_metrics["probs"],
                        cfg["class_names"], epoch, cfg["results_dir"])
        plot_per_class_metrics(v_metrics, cfg["class_names"],
                               epoch, cfg["results_dir"])

    # ── final summary plots ───────────────────────────────────────────────────
    plot_training_curves(tracker, cfg["results_dir"])

    # ── restore best weights ──────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    final_path = os.path.join(cfg["checkpoint_dir"], "final_best_model.pth")
    torch.save({"model_state": best_weights, "config": cfg}, final_path)
    logger.info("=" * 70)
    logger.info("Training complete. Best val loss: %.5f", best_val_loss)
    logger.info("Final model saved → %s", final_path)


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def load_and_predict(
    checkpoint_path: str,
    image_path: str,
    cfg: dict,
    device: torch.device | None = None,
) -> dict:
    """
    Load a saved checkpoint and run inference on a single retinal image.

    Returns a dict with predicted class name and per-class probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # preprocess
    from PIL import Image
    tfm   = get_transforms(cfg["image_size"], "val")
    img   = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    # forward
    with torch.no_grad():
        logits = model(img)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred   = int(np.argmax(probs))

    return {
        "predicted_class": cfg["class_names"][pred],
        "class_index"    : pred,
        "probabilities"  : {name: float(p)
                            for name, p in zip(cfg["class_names"], probs)},
    }


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(CONFIG)
