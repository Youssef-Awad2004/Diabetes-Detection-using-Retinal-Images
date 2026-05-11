"""
evaluation/metrics.py
─────────────────────
Computes and logs all clinical evaluation metrics:
  - Accuracy
  - Macro AUC-ROC  (one-vs-rest)
  - Per-class Sensitivity (Recall) and Specificity
  - Confusion matrix
  - Full sklearn classification report
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    preds:  np.ndarray,
    probs:  np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Compute the full set of clinical metrics for a single evaluation pass.

    Sensitivity and Specificity are derived one-vs-rest from the confusion
    matrix — standard practice in ophthalmology and medical imaging literature.

    Parameters
    ----------
    labels       : ground-truth integer class indices  (N,)
    preds        : argmax predictions                  (N,)
    probs        : softmax probabilities               (N, C)
    class_names  : list of human-readable class labels

    Returns
    -------
    dict with keys:
      accuracy, auc_roc, macro_sensitivity, macro_specificity,
      sensitivity_per_class, specificity_per_class,
      confusion_matrix, labels_bin, probs
    """
    num_classes = len(class_names)
    accuracy    = accuracy_score(labels, preds)

    # ── Macro AUC-ROC (one-vs-rest) ───────────────────────────────────────────
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    try:
        auc_macro = roc_auc_score(
            labels_bin, probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        # Can happen when a class is absent from the current val batch
        auc_macro = float("nan")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # ── Per-class Sensitivity & Specificity from CM (one-vs-rest) ─────────────
    sensitivity_per_class = []
    specificity_per_class = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP          # missed positives
        FP = cm[:, i].sum() - TP          # false alarms
        TN = cm.sum() - TP - FN - FP      # true negatives

        sensitivity_per_class.append(TP / (TP + FN + 1e-8))
        specificity_per_class.append(TN / (TN + FP + 1e-8))

    return {
        "accuracy"              : accuracy,
        "auc_roc"               : auc_macro,
        "macro_sensitivity"     : float(np.mean(sensitivity_per_class)),
        "macro_specificity"     : float(np.mean(specificity_per_class)),
        "sensitivity_per_class" : sensitivity_per_class,
        "specificity_per_class" : specificity_per_class,
        "confusion_matrix"      : cm,
        "labels_bin"            : labels_bin,   # kept for ROC curve plotting
        "probs"                 : probs,         # kept for ROC curve plotting
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_metrics(
    phase: str,
    epoch: int,
    loss:  float,
    metrics: dict,
    class_names: list[str],
) -> None:
    """
    Print a compact summary line plus per-class sensitivity / specificity.
    Silently skips per-class lines when values are NaN (e.g. train phase).
    """
    logger.info(
        "[%s] Epoch %3d | Loss: %.4f | Acc: %.4f | AUC: %.4f | "
        "Sens: %.4f | Spec: %.4f",
        phase, epoch, loss,
        metrics["accuracy"],
        metrics["auc_roc"],
        metrics["macro_sensitivity"],
        metrics["macro_specificity"],
    )

    # Per-class breakdown (only meaningful for val/test)
    sens_list = metrics.get("sensitivity_per_class", [])
    spec_list = metrics.get("specificity_per_class", [])
    for i, cls in enumerate(class_names):
        if i < len(sens_list):
            logger.info(
                "         %-22s  Sensitivity: %.3f   Specificity: %.3f",
                cls, sens_list[i], spec_list[i],
            )


def print_classification_report(
    labels: np.ndarray,
    preds:  np.ndarray,
    class_names: list[str],
) -> None:
    """Print sklearn's full per-class precision / recall / F1 report."""
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    logger.info("\n%s", report)
