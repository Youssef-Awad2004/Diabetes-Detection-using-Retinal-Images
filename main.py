"""
main.py
───────
Entry point for the Diabetic Retinopathy grading pipeline.

Wires together all modules and launches training:
  config.py              — hyper-parameters & paths
  data/dataset.py        — transforms, datasets, dataloaders
  models/model.py        — EfficientNet-B3 / ResNet-50 backbone + head
  training/trainer.py    — optimiser, scheduler, train/val loops, checkpointing
  evaluation/metrics.py  — accuracy, AUC-ROC, sensitivity, specificity
  evaluation/plots.py    — confusion matrix, ROC curves, training curves
  utils/utils.py         — seeding, device selection

Usage
─────
  python main.py

Edit CONFIG in config.py before running.
"""

import logging
from pathlib import Path

from config import CONFIG
from data.dataset import build_dataloaders
from models.model import build_model
from training.trainer import train
from evaluation.metrics import compute_metrics, log_metrics, print_classification_report
from evaluation.plots import plot_epoch_results, plot_training_curves
from utils.utils import seed_everything, get_device

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  EPOCH CALLBACK
#  Wraps the per-epoch plotting + report printing into a single callable
#  that the trainer calls without needing to import evaluation modules itself.
# ─────────────────────────────────────────────────────────────────────────────

def epoch_plot_callback(metrics, labels, preds, probs, class_names, epoch, results_dir):
    plot_epoch_results(metrics, labels, preds, probs, class_names, epoch, results_dir)
    print_classification_report(labels, preds, class_names)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = CONFIG

    # ── reproducibility & device ──────────────────────────────────────────────
    seed_everything(cfg["seed"])
    device = get_device()

    # ── create output directories ─────────────────────────────────────────────
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    logger.info("Loading datasets …")
    train_loader, val_loader, class_weights = build_dataloaders(cfg)

    # ── model ─────────────────────────────────────────────────────────────────
    logger.info("Building model: %s", cfg["model_name"])
    model = build_model(cfg).to(device)

    # ── train ─────────────────────────────────────────────────────────────────
    model, tracker = train(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        class_weights  = class_weights,
        cfg            = cfg,
        device         = device,
        # inject evaluation callables (keeps trainer.py decoupled)
        compute_metrics_fn = compute_metrics,
        log_metrics_fn     = log_metrics,
        plot_epoch_fn      = epoch_plot_callback,
        plot_curves_fn     = plot_training_curves,
    )

    logger.info("Pipeline complete. Outputs saved to '%s' and '%s'.",
                cfg["results_dir"], cfg["checkpoint_dir"])


if __name__ == "__main__":
    main()
