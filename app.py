from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import base64
import numpy as np
import torch
from io import BytesIO
from PIL import Image
import os

from config import CONFIG
from dataset import get_transforms
from model import build_model
from utils import get_device

app = FastAPI()

# Define the request model for base64 input
class ImageRequest(BaseModel):
    image: str  # base64 encoded image

# ── Model Loading ────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/final_best_model.pth")
device = get_device()
model = None
transforms = None

def load_model():
    """Load the model and transforms on startup."""
    global model, transforms
    try:
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")
            return False
        
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model = build_model(CONFIG).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        
        transforms = get_transforms(CONFIG["image_size"], "val")
        print(f"Model loaded successfully from {CHECKPOINT_PATH}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def process_image(image: Image.Image) -> dict:
    """
    Run inference on the image and return predictions.
    Returns:
        dict with predicted_class, class_index, and probabilities
    """
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Convert PIL image to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply transforms and create batch
        tensor = transforms(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred = int(np.argmax(probs))
        
        # Format results
        result = {
            "predicted_class": CONFIG["class_names"][pred],
            "class_index": pred,
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(CONFIG["class_names"], probs)
            },
        }
        return result
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Load model on app startup."""
    load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    POST endpoint that accepts an image file and returns diabetes retinopathy prediction.
    Use form-data in Postman: key="file", value=<select image file>
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Run inference
        result = process_image(image)
        
        return result
    
    except Exception as e:
        return {"error": str(e), "file_name": file.filename}

@app.post("/upload")
async def upload(file: UploadFile = File(None)) -> dict:
    """
    Alternative POST endpoint - more lenient file handling.
    """
    if not file:
        return {"error": "No file provided"}
    
    try:
        contents = await file.read()
        if not contents:
            return {"error": "File is empty"}
        
        image = Image.open(BytesIO(contents))
        result = process_image(image)
        
        return {**result, "filename": file.filename}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict-base64")
async def predict_base64(request: ImageRequest) -> dict:
    """
    POST endpoint that accepts a base64 encoded image in JSON
    and returns prediction.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_data))
        
        # Run inference
        result = process_image(image)
        
        return result
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
