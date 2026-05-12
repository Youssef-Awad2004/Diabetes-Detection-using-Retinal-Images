"""
inference.py
────────────
Two modes:

  1. Batch submission (default) — runs the trained model over every image in
     test.csv and writes a Kaggle-ready submission.csv.

       python inference.py --checkpoint checkpoints/best_model.pth

  2. Single-image — predicts the DR grade for one image and prints probabilities.

       python inference.py --checkpoint checkpoints/best_model.pth \
                           --image path/to/retinal_image.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import CONFIG
from dataset import get_transforms
from model import build_model
from utils import get_device


# ─────────────────────────────────────────────────────────────────────────────
#  TEST DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    """
    Wraps test.csv + test_images/ for batch inference.
    Has no labels — returns (image_tensor, id_code) pairs.
    """

    def __init__(self, test_csv: str, images_dir: str, transform):
        self.df         = pd.read_csv(test_csv)
        self.images_dir = images_dir
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        id_code  = self.df.iloc[idx]["id_code"]
        img_path = os.path.join(self.images_dir, id_code + ".png")
        img      = Image.open(img_path).convert("RGB")
        return self.transform(img), id_code


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH INFERENCE  →  submission.csv
# ─────────────────────────────────────────────────────────────────────────────

def generate_submission(checkpoint_path: str, cfg: dict) -> str:
    """
    Run the trained model over all images in test.csv and save submission.csv.

    Returns the path to the saved submission file.
    """
    device = get_device()

    # ── load model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── test dataloader ───────────────────────────────────────────────────────
    tfm     = get_transforms(cfg["image_size"], "val", cfg)
    dataset = TestDataset(cfg["test_csv"], cfg["test_images_dir"], tfm)
    loader  = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # ── forward pass over all test images ────────────────────────────────────
    all_ids, all_preds = [], []
    with torch.no_grad():
        for images, ids in tqdm(loader, desc="Generating predictions"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_ids.extend(ids)
            all_preds.extend(preds.tolist())

    # ── write submission ──────────────────────────────────────────────────────
    os.makedirs(cfg["results_dir"], exist_ok=True)
    submission_path = os.path.join(cfg["results_dir"], "submission.csv")
    submission_df   = pd.DataFrame({"id_code": all_ids, "diagnosis": all_preds})
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission saved → {submission_path}")
    print(f"Predictions: {submission_df['diagnosis'].value_counts().sort_index().to_dict()}")
    return submission_path


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-IMAGE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(checkpoint_path: str, image_path: str, cfg: dict) -> dict:
    """
    Run inference on a single image and return the predicted grade + probabilities.
    """
    device = get_device()

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm    = get_transforms(cfg["image_size"], "val", cfg)
    img    = Image.open(image_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred   = int(np.argmax(probs))

    return {
        "predicted_class": cfg["class_names"][pred],
        "class_index"    : pred,
        "probabilities"  : {
            name: round(float(p), 4)
            for name, p in zip(cfg["class_names"], probs)
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR grading inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--image", default=None,
        help="Path to a single retinal image. Omit to run batch submission over test.csv.",
    )
    args = parser.parse_args()

    if args.image:
        # ── single-image mode ─────────────────────────────────────────────────
        result = predict_single(args.checkpoint, args.image, CONFIG)
        print(f"\nPredicted grade : {result['predicted_class']}  (class {result['class_index']})")
        print("\nPer-class probabilities:")
        for cls, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"  {cls:<22} {prob:.4f}  {bar}")
    else:
        # ── batch submission mode ─────────────────────────────────────────────
        generate_submission(args.checkpoint, CONFIG)
