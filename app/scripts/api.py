#!/usr/bin/env python3
"""
ECOSORT AI API — Complete with OpenCV Preprocessing
====================================================
Save as: app/scripts/api.py
Run from: app/scripts/ folder
Command:  py -3.12 -m uvicorn api:app --host 0.0.0.0 --port 8000
"""
import os
import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ========== CONFIG ==========
# AFTER
MODEL_PATH = "best_model_final.keras"
CLASS_NAMES_PATH = "class_names.txt"
# ========== LOAD MODEL ==========
print("Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run from the folder containing the model.")

model = tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names not found: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]

print(f"✅ Model loaded. Classes: {CLASS_NAMES}")

# ========== DISPOSAL INFO ==========
DISPOSAL = {
    "plastic": {"bin": "Blue Recycling Bin", "tip": "Rinse and crush to save space."},
    "paper": {"bin": "Green Recycling Bin", "tip": "Keep dry. No greasy pizza boxes."},
    "cardboard": {"bin": "Green Recycling Bin", "tip": "Flatten boxes before recycling."},
    "glass": {"bin": "Red Recycling Bin", "tip": "Rinse and remove lids."},
    "metal": {"bin": "Purple Recycling Bin", "tip": "Rinse cans. Labels okay."},
    "organic": {"bin": "Brown Compost Bin", "tip": "No meat or dairy in home compost."},
    "trash": {"bin": "Black Landfill Bin", "tip": "Cannot be recycled. Dispose safely."},
}

# ========== FASTAPI APP ==========
app = FastAPI(title="EcoSort AI", version="1.0")

# ========== PREPROCESSING ==========
def auto_rotate(image_bytes):
    """Fix phone camera rotation using EXIF."""
    img = Image.open(io.BytesIO(image_bytes))
    try:
        exif = img._getexif()
        if exif:
            orientation = exif.get(274, 1)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return np.array(img)

def smart_crop(img, padding=0.1):
    """Crop to main object using edge detection."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[top:top+min_dim, left:left+min_dim]
    
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    pad_x, pad_y = int(cw * padding), int(ch * padding)
    
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(w, x + cw + pad_x), min(h, y + ch + pad_y)
    cropped = img[y1:y2, x1:x2]
    
    if cropped.shape[0] < 50 or cropped.shape[1] < 50:
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[top:top+min_dim, left:left+min_dim]
    
    return cropped

def preprocess_image(image_bytes, target_size=300):
    """
    Full pipeline: rotate → smart crop → resize → RGB → batch
    Output: [1, 300, 300, 3] float32, pixels 0-255 (NO /255!)
    """
    # Fix phone rotation
    img = auto_rotate(image_bytes)
    
    # Convert to BGR for OpenCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Smart crop to object
    img = smart_crop(img)
    
    # Resize to model input
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Batch + float32
    return np.expand_dims(img, axis=0).astype(np.float32)

def predict_image(image_bytes):
    """Run prediction."""
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array, verbose=0)[0]
    
    top3_idx = np.argsort(preds)[-3:][::-1]
    top1_idx = top3_idx[0]
    confidence = float(preds[top1_idx])
    gap = float(preds[top1_idx] - preds[top3_idx[1]])
    
    return {
        "prediction": CLASS_NAMES[top1_idx],
        "confidence": round(confidence * 100, 2),
        "reliable": confidence > 0.75 and gap > 0.15,
        "top_3": [
            {"class": CLASS_NAMES[i], "confidence": round(float(preds[i]) * 100, 2)}
            for i in top3_idx
        ],
        "disposal": DISPOSAL.get(CLASS_NAMES[top1_idx], {"bin": "Unknown", "tip": "Check local guidelines."}),
    }

# ========== API ENDPOINTS ==========
@app.get("/")
def root():
    return {
        "message": "EcoSort AI API is running",
        "version": "1.0",
        "classes": CLASS_NAMES,
        "model_input": "RGB image, 300x300, pixel values 0-255"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload image → get classification + disposal info."""
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "File must be an image"})
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        return JSONResponse(status_code=400, content={"error": "Image too large (max 10MB)"})
    
    try:
        result = predict_image(contents)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ========== FALLBACK: RUN DIRECTLY ==========
if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)