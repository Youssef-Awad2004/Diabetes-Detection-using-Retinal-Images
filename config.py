"""
config.py
─────────
Central configuration for the DR grading pipeline.
Edit values here; every other module imports from this file.
"""

CONFIG = {
    # ── Paths ────────────────────────────────────────────────────────────────
    "train_dir"      : "data/train",
    "val_dir"        : "data/val",
    "checkpoint_dir" : "checkpoints",
    "results_dir"    : "results",

    # ── Model ────────────────────────────────────────────────────────────────
    "num_classes"    : 5,
    "model_name"     : "efficientnet_b3",   # or "resnet50"
    "freeze_fraction": 0.7,                 # fraction of backbone blocks to freeze

    # ── Training ─────────────────────────────────────────────────────────────
    "epochs"         : 40,
    "batch_size"     : 32,
    "num_workers"    : 4,
    "image_size"     : 300,

    # ── Optimiser ────────────────────────────────────────────────────────────
    "lr"             : 3e-4,
    "weight_decay"   : 1e-4,
    "lr_patience"    : 5,
    "lr_factor"      : 0.3,
    "min_lr"         : 1e-7,
    "label_smoothing": 0.1,

    # ── Misc ─────────────────────────────────────────────────────────────────
    "seed"           : 42,
    "class_names"    : [
        "Normal", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"
    ],
}
