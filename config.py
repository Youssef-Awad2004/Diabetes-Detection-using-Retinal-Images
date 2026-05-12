"""
config.py
─────────
Central configuration for the DR grading pipeline.
Edit values here; every other module imports from this file.
"""

CONFIG = {
    # ── Paths ────────────────────────────────────────────────────────────────
    # APTOS / Kaggle flat-folder format:
    #   train.csv        → id_code, diagnosis (0-4)
    #   train_images/    → <id_code>.png  (no class subdirectories)
    #   test.csv         → id_code only (unlabelled Kaggle test set)
    #   test_images/     → <id_code>.png
    "train_csv"        : "data/train.csv",
    "train_images_dir" : "data/train_images",
    "test_csv"         : "data/test.csv",
    "test_images_dir"  : "data/test_images",
    "checkpoint_dir"   : "checkpoints",
    "results_dir"      : "results",

    # Stratified 80/10/10 split of train.csv into train / val / test.
    # val   — used for LR scheduling & early stopping (indirectly touched)
    # test  — held-out, never seen during training; final unbiased evaluation
    # test_images/ remains for the unlabelled Kaggle submission only.
    "val_split"        : 0.1,
    "test_split"       : 0.1,

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

    # ── Preprocessing (quality-aware, EDA-driven) ─────────────────────────
    # Ben Graham: circle-crop + local-average subtraction.
    # Removes camera lighting gradient; eliminates R/G intensity confound.
    "use_ben_graham"  : True,

    # CLAHE: adaptive contrast normalisation on LAB L-channel.
    # Breaks the sharpness confound (r=-0.517, p<0.001).
    "use_clahe"       : True,
    "clahe_clip_limit": 2.0,    # higher = more aggressive contrast boost
    "clahe_tile_grid" : (8, 8), # finer grid = more localised equalisation

    # WeightedRandomSampler: fixes imbalance at the batch level.
    # Each epoch draws samples proportional to inverse class frequency,
    # ensuring all 5 DR grades are equally represented in every batch.
    "use_weighted_sampler": True,

    # MixUp: interpolates images+labels between two samples per batch.
    # Teaches the model the ordinal severity continuum (EDA Finding #2).
    # alpha=0.4 is conservative — strong enough to regularise without
    # destroying fine lesion signals critical for DR grading.
    # Set to 0.0 to disable MixUp entirely.
    "mixup_alpha"     : 0.4,
}
