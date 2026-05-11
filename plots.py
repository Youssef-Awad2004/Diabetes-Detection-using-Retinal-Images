"""
evaluation/plots.py
───────────────────
All Matplotlib / Seaborn visualisations:
  - Confusion matrix (raw + normalised)
  - ROC curves (per-class + macro average)
  - Per-class sensitivity & specificity bar chart
  - Training / validation curves (loss, accuracy, AUC, sens/spec)

Every function saves a PNG to results_dir and closes the figure to avoid
memory leaks during long training runs.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")   # headless — works on servers without a display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    epoch: int,
    results_dir: str,
) -> None:
    """Save a side-by-side raw-counts + row-normalised confusion matrix."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix — Epoch {epoch}", fontweight="bold")

    for ax, data, fmt, title in [
        (ax1, cm,      "d",    "Raw Counts"),
        (ax2, cm_norm, ".2f",  "Normalised (row %)"),
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
    _save(fig, results_dir, f"confusion_matrix_epoch{epoch:03d}.png")


# ─────────────────────────────────────────────────────────────────────────────
#  ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    labels_bin: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    epoch: int,
    results_dir: str,
) -> None:
    """
    One-vs-rest ROC curve per class plus the macro-average curve.
    labels_bin must be binarised with sklearn.preprocessing.label_binarize.
    """
    num_classes = len(class_names)
    palette     = plt.cm.get_cmap("tab10", num_classes)
    mean_fpr    = np.linspace(0, 1, 200)
    tprs        = []

    fig, ax = plt.subplots(figsize=(9, 7))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=palette(i), lw=1.8,
                label=f"{class_names[i]}  (AUC = {roc_auc_val:.3f})")
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr  = np.mean(tprs, axis=0)
    macro_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color="black", lw=2.5, linestyle="--",
            label=f"Macro Average  (AUC = {macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curves (OvR) — Epoch {epoch}", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    _save(fig, results_dir, f"roc_curves_epoch{epoch:03d}.png")


# ─────────────────────────────────────────────────────────────────────────────
#  PER-CLASS SENSITIVITY & SPECIFICITY
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_metrics(
    metrics: dict,
    class_names: list[str],
    epoch: int,
    results_dir: str,
) -> None:
    """Grouped bar chart of per-class sensitivity and specificity."""
    x     = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_sens = ax.bar(x - width / 2, metrics["sensitivity_per_class"],
                       width, label="Sensitivity", color="steelblue")
    bars_spec = ax.bar(x + width / 2, metrics["specificity_per_class"],
                       width, label="Specificity", color="darkorange")

    # value labels on each bar
    for bar in (*bars_sens, *bars_spec):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Per-Class Sensitivity & Specificity — Epoch {epoch}",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, results_dir, f"per_class_metrics_epoch{epoch:03d}.png")


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING CURVES  (called once after all epochs)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(tracker, results_dir: str) -> None:
    """
    2×2 grid of epoch-level curves:
      Loss | Accuracy | AUC-ROC | Sensitivity & Specificity
    tracker is a MetricsTracker instance from training/trainer.py.
    """
    epochs = range(1, len(tracker.train_loss) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves — Diabetic Retinopathy Grading",
                 fontsize=14, fontweight="bold")

    # (ax, title, train_series, val_series, train_label, val_label, t_color, v_color)
    configs = [
        (axes[0, 0], "Loss",     tracker.train_loss, tracker.val_loss,
         "Train",        "Val",         "royalblue",  "crimson"),
        (axes[0, 1], "Accuracy", tracker.train_acc,  tracker.val_acc,
         "Train",        "Val",         "seagreen",   "darkorange"),
        (axes[1, 0], "AUC-ROC",  None,               tracker.val_auc,
         None,           "Val AUC-ROC", None,         "purple"),
        (axes[1, 1], "Sens / Spec", None,             tracker.val_sensitivity,
         None,           "Sensitivity", None,         "teal"),
    ]

    for ax, title, train_data, val_data, tl, vl, tc, vc in configs:
        if train_data is not None:
            ax.plot(epochs, train_data, color=tc, label=tl, linewidth=2)
        ax.plot(epochs, val_data, color=vc, label=vl, linewidth=2, linestyle="--")
        if title == "Sens / Spec":
            ax.plot(epochs, tracker.val_specificity,
                    color="coral", label="Specificity", linewidth=2, linestyle=":")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, results_dir, "training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
#  CONVENIENCE WRAPPER — called from the training loop once per epoch
# ─────────────────────────────────────────────────────────────────────────────

def plot_epoch_results(
    metrics: dict,
    labels: np.ndarray,
    preds:  np.ndarray,
    probs:  np.ndarray,
    class_names: list[str],
    epoch: int,
    results_dir: str,
) -> None:
    """
    Call all three per-epoch plots in one shot.
    Imported and wired up in main.py.
    """
    os.makedirs(results_dir, exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, epoch, results_dir)
    plot_roc_curves(metrics["labels_bin"], metrics["probs"], class_names, epoch, results_dir)
    plot_per_class_metrics(metrics, class_names, epoch, results_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, results_dir: str, filename: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", out)
