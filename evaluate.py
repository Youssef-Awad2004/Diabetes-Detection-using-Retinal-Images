"""
evaluate.py
───────────
Run final metrics on any saved checkpoint against the held-out test split —
without re-running the full training pipeline.

Usage
─────
  # Use the final best model (default)
  python evaluate.py

  # Use a specific epoch checkpoint
  python evaluate.py --checkpoint checkpoints/best_model_epoch008.pth
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import build_dataloaders
from model import build_model
from metrics import compute_metrics, log_metrics, print_classification_report
from plots import plot_epoch_results
from utils import seed_everything, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate(checkpoint_path: str) -> None:
    cfg = CONFIG
    seed_everything(cfg["seed"])
    device = get_device()

    # ── data: only the test split is needed ───────────────────────────────────
    # build_dataloaders applies the same deterministic seed-based split used
    # during training, so the test set is identical.
    logger.info("Reconstructing held-out test split …")
    _, _, test_loader, class_weights = build_dataloaders(cfg)

    # ── model ─────────────────────────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)

    # both epoch checkpoints (key="model_state") and final_best_model (same key)
    state_key = "model_state" if "model_state" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    saved_epoch    = ckpt.get("epoch",    0)
    saved_val_loss = ckpt.get("val_loss", float("nan"))
    saved_epoch    = int(saved_epoch) if str(saved_epoch).isdigit() else 0
    logger.info("  Saved at epoch %s  |  val_loss=%.5f", saved_epoch, saved_val_loss)

    # ── forward pass over test set ────────────────────────────────────────────
    criterion = CrossEntropyLoss(weight=class_weights.to(device))

    all_labels, all_preds, all_probs_list = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            import torch.nn.functional as F
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs_list.append(probs)

    import numpy as np
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.vstack(all_probs_list)
    test_loss  = running_loss / len(test_loader.dataset)

    # ── metrics ───────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("FINAL EVALUATION on held-out test set")
    logger.info("=" * 70)

    test_metrics = compute_metrics(all_labels, all_preds, all_probs, cfg["class_names"])
    log_metrics("TEST", saved_epoch, test_loss, test_metrics, cfg["class_names"])
    print_classification_report(all_labels, all_preds, cfg["class_names"])

    # ── save confusion matrix / ROC / per-class plot ──────────────────────────
    # plot_epoch_results uses epoch:03d formatting, so pass a plain int.
    # For final_best_model the epoch key is the best training epoch.
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    plot_epoch_results(
        test_metrics, all_labels, all_preds, all_probs,
        cfg["class_names"],
        epoch=int(saved_epoch) if str(saved_epoch).isdigit() else 999,
        results_dir=cfg["results_dir"],
    )
    logger.info("Plots saved to '%s'.", cfg["results_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved DR model on the held-out test set.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/final_best_model.pth",
        help="Path to a .pth checkpoint file (default: checkpoints/final_best_model.pth)",
    )
    args = parser.parse_args()
    evaluate(args.checkpoint)
