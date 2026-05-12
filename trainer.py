"""
training/trainer.py
───────────────────
Owns the full training lifecycle:
  - Loss function, optimiser, and LR scheduler construction
  - Single-epoch train / val loop  (run_epoch)
  - Full training loop with checkpointing and metric tracking
"""

import copy
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils import mixup_batch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Alpha-balanced focal loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 down-weights easy, well-classified examples so the gradient
    focuses on the hard, misclassified ones — critical for Moderate/Severe DR
    which are consistently ignored under plain cross-entropy.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        # Use CE with reduction='none' so we can apply the focal modulator per sample
        self._ce = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
            reduction="none",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # pt must be the raw (unweighted) probability of the ground-truth class.
        # Using exp(-weighted_ce) as a proxy is incorrect: it computes p_c^weight,
        # which collapses to ~0 for minority classes and kills the focal modulation.
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)                          # (N, C)
            pt    = probs.gather(1, targets.view(-1, 1)).squeeze(1)   # (N,)
        focal_weight = (1.0 - pt) ** self.gamma                       # (N,)

        ce_loss = self._ce(logits, targets)          # (N,)  weighted + smoothed
        return (focal_weight * ce_loss).mean()


# ─────────────────────────────────────────────────────────────────────────────
#  EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when a monitored metric stops improving.

    Parameters
    ----------
    patience  : epochs to wait after last improvement before stopping
    min_delta : minimum change to qualify as an improvement
    mode      : 'min' (loss) or 'max' (accuracy / AUC)
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.triggered = False

    def step(self, metric: float) -> bool:
        """Call once per epoch. Returns True when training should stop."""
        improved = (
            (self.mode == "min" and metric < self.best - self.min_delta)
            or (self.mode == "max" and metric > self.best + self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping: no improvement for %d / %d epochs (best=%.5f, current=%.5f)",
                self.counter, self.patience, self.best, metric,
            )
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMISER & SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: dict,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Constructs:
      criterion  — CrossEntropyLoss with label smoothing + class weights
      optimizer  — AdamW; backbone params use 0.1× the head learning rate
      scheduler  — ReduceLROnPlateau watching val loss

    The differential learning-rate trick prevents catastrophic forgetting:
    pre-trained backbone weights are updated conservatively while the new
    classification head is trained at full speed.
    """
    # ── loss ──────────────────────────────────────────────────────────────────
    gamma = cfg.get("focal_loss_gamma", 0.0)
    if gamma > 0.0:
        criterion = FocalLoss(
            weight=class_weights.to(device),
            gamma=gamma,
            label_smoothing=cfg["label_smoothing"],
        )
        logger.info("Loss: FocalLoss (gamma=%.1f, label_smoothing=%.2f)", gamma, cfg["label_smoothing"])
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=cfg["label_smoothing"],
        )
        logger.info("Loss: CrossEntropyLoss (label_smoothing=%.2f)", cfg["label_smoothing"])

    # ── split params: backbone vs head ────────────────────────────────────────
    backbone_params, head_params = [], []
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
    )

    return criterion, optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    phase: str,
    mixup_alpha: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one epoch of training or validation.

    Parameters
    ----------
    model       : the network
    loader      : DataLoader for the current phase
    criterion   : loss function
    optimizer   : pass None for validation (no backward pass)
    device      : torch.device
    phase       : "train" | "val"
    mixup_alpha : Beta distribution alpha for MixUp.  0 = disabled.
                  Applied only during training; ignored for validation.

    Returns
    -------
    avg_loss   : mean cross-entropy over the full loader
    all_labels : ground-truth integer labels  (N,)
    all_preds  : argmax predictions           (N,)
    all_probs  : softmax probabilities        (N, C)
    """
    is_train = phase == "train"
    use_mixup = is_train and mixup_alpha > 0.0
    model.train() if is_train else model.eval()

    running_loss            = 0.0
    all_labels, all_preds, all_probs = [], [], []

    pbar = tqdm(
        loader,
        desc=f"{phase.upper():<5}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_mixup:
                images, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=mixup_alpha)
                logits = model(images)
                loss   = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                # For metric logging, report the dominant label
                labels = labels_a if lam >= 0.5 else labels_b
            else:
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping — prevents exploding gradients on small batches
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            probs  = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds  = np.argmax(probs, axis=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)

            # live loss in the progress bar suffix
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss   = running_loss / len(loader.dataset)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.vstack(all_probs)

    return avg_loss, all_labels, all_preds, all_probs


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT
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
    Persist model weights + optimizer state whenever val loss improves.
    Saves to  <checkpoint_dir>/best_model_epoch<NNN>.pth
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_loss"   : val_loss,
        "config"     : cfg,
    }
    path = f"{checkpoint_dir}/best_model_epoch{epoch:03d}.pth"
    torch.save(checkpoint, path)
    logger.info("✓ Checkpoint saved → %s  (val_loss=%.5f)", path, val_loss)


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS TRACKER  (passed to evaluation module for plotting)
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Accumulates per-epoch scalar metrics for end-of-training curve plots."""

    def __init__(self):
        self.train_loss           : list[float] = []
        self.val_loss             : list[float] = []
        self.train_acc            : list[float] = []
        self.val_acc              : list[float] = []
        self.val_balanced_acc     : list[float] = []
        self.val_auc              : list[float] = []
        self.val_weighted_auc     : list[float] = []
        self.val_macro_f1         : list[float] = []
        self.val_macro_precision  : list[float] = []
        self.val_macro_recall     : list[float] = []
        self.val_sensitivity      : list[float] = []
        self.val_specificity      : list[float] = []

    def update(
        self,
        t_loss: float,
        t_acc:  float,
        v_loss: float,
        v_metrics: dict,
    ) -> None:
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.train_acc.append(t_acc)
        self.val_acc.append(v_metrics["accuracy"])
        self.val_balanced_acc.append(v_metrics.get("balanced_accuracy", float("nan")))
        self.val_auc.append(v_metrics["auc_roc"])
        self.val_weighted_auc.append(v_metrics.get("weighted_auc_roc", float("nan")))
        self.val_macro_f1.append(v_metrics.get("macro_f1", float("nan")))
        self.val_macro_precision.append(v_metrics.get("macro_precision", float("nan")))
        self.val_macro_recall.append(v_metrics.get("macro_recall", float("nan")))
        self.val_sensitivity.append(v_metrics["macro_sensitivity"])
        self.val_specificity.append(v_metrics["macro_specificity"])


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    class_weights: torch.Tensor,
    cfg: dict,
    device: torch.device,
    compute_metrics_fn,      # callable from evaluation/metrics.py
    log_metrics_fn,          # callable from evaluation/metrics.py
    plot_epoch_fn,           # callable from evaluation/plots.py
    plot_curves_fn,          # callable from evaluation/plots.py
) -> tuple[nn.Module, MetricsTracker]:
    """
    Full training loop.

    External callables are injected so this module stays decoupled from the
    evaluation and plotting layers.

    Returns the model (loaded with best weights) and the MetricsTracker.
    """
    criterion, optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, class_weights, device
    )

    tracker       = MetricsTracker()
    best_val_loss = float("inf")
    best_weights  = copy.deepcopy(model.state_dict())

    es_patience = cfg.get("early_stopping_patience", 0)
    early_stopper = EarlyStopping(
        patience=es_patience,
        min_delta=cfg.get("early_stopping_min_delta", 1e-4),
        mode="min",
    ) if es_patience > 0 else None

    logger.info("=" * 70)
    logger.info(
        "Starting training — %d epochs%s",
        cfg["epochs"],
        f"  |  EarlyStopping patience={es_patience}" if es_patience > 0 else "",
    )
    logger.info("=" * 70)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # ── train phase ───────────────────────────────────────────────────────
        t_loss, t_labels, t_preds, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, "train",
            mixup_alpha=cfg.get("mixup_alpha", 0.0),
        )
        t_acc = accuracy_score(t_labels, t_preds)

        # ── val phase ─────────────────────────────────────────────────────────
        v_loss, v_labels, v_preds, v_probs = run_epoch(
            model, val_loader, criterion, None, device, "val",
            mixup_alpha=0.0,  # never apply MixUp during evaluation
        )
        v_metrics = compute_metrics_fn(v_labels, v_preds, v_probs, cfg["class_names"])

        # ── LR schedule ───────────────────────────────────────────────────────
        scheduler.step(v_loss)

        # ── logging ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        logger.info("── Epoch %d/%d  (%.1fs) ─────────────────", epoch, cfg["epochs"], elapsed)
        log_metrics_fn("TRAIN", epoch, t_loss,
                       {"accuracy": t_acc, "balanced_accuracy": float("nan"),
                        "auc_roc": float("nan"), "weighted_auc_roc": float("nan"),
                        "macro_f1": float("nan"), "macro_precision": float("nan"),
                        "macro_recall": float("nan"),
                        "macro_sensitivity": float("nan"),
                        "macro_specificity": float("nan")},
                       cfg["class_names"])
        log_metrics_fn("VAL", epoch, v_loss, v_metrics, cfg["class_names"])

        # ── checkpointing — save when val loss improves ───────────────────────
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights  = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, epoch, v_loss, cfg, cfg["checkpoint_dir"])

        # ── update tracker ────────────────────────────────────────────────────
        tracker.update(t_loss, t_acc, v_loss, v_metrics)

        # ── per-epoch plots ───────────────────────────────────────────────────
        plot_epoch_fn(v_metrics, v_labels, v_preds, v_probs,
                      cfg["class_names"], epoch, cfg["results_dir"])

        # ── early stopping ────────────────────────────────────────────────────
        if early_stopper is not None and early_stopper.step(v_loss):
            logger.info(
                "Early stopping triggered at epoch %d — no improvement for %d epochs "
                "(best val_loss=%.5f).",
                epoch, early_stopper.patience, early_stopper.best,
            )
            break

    # ── end of training ───────────────────────────────────────────────────────
    plot_curves_fn(tracker, cfg["results_dir"])

    # restore and persist the overall best weights
    model.load_state_dict(best_weights)
    final_path = f"{cfg['checkpoint_dir']}/final_best_model.pth"
    import torch as _torch
    _torch.save({"model_state": best_weights, "config": cfg}, final_path)
    logger.info("Best val loss: %.5f — final model saved → %s", best_val_loss, final_path)

    return model, tracker
