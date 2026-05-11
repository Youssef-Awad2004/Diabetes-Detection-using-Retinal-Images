"""
inference.py
────────────
Run a trained checkpoint on a single retinal image and print the
DR grade with per-class probabilities.

Usage
─────
  python inference.py --checkpoint checkpoints/final_best_model.pth \
                      --image path/to/retinal_image.jpg
"""

import argparse
import numpy as np
import torch
from PIL import Image

from config import CONFIG
from data.dataset import get_transforms
from models.model import build_model
from utils.utils import get_device


def predict(checkpoint_path: str, image_path: str, cfg: dict) -> dict:
    """
    Load a saved checkpoint and run inference on a single image.

    Returns
    -------
    dict with keys:
      predicted_class  : human-readable class name
      class_index      : integer 0-4
      probabilities    : {class_name: probability, …}
    """
    device = get_device()

    # ── load model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── preprocess ────────────────────────────────────────────────────────────
    tfm   = get_transforms(cfg["image_size"], "val")
    img   = Image.open(image_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0).to(device)

    # ── forward ───────────────────────────────────────────────────────────────
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR grading inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image",      required=True, help="Path to retinal image")
    args = parser.parse_args()

    result = predict(args.checkpoint, args.image, CONFIG)

    print(f"\nPredicted grade : {result['predicted_class']}  (class {result['class_index']})")
    print("\nPer-class probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {cls:<22} {prob:.4f}  {bar}")
