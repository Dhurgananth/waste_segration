#!/usr/bin/env python3
"""
PHOTO PREPROCESSING PIPELINE
============================
Run this BEFORE model.predict() on every user photo.
"""
import cv2
import numpy as np
from PIL import Image
import io

def auto_rotate(image_bytes):
    """
    Fix phone camera rotation using EXIF data.
    Phones save portrait photos as landscape + EXIF rotation tag.
    """
    img = Image.open(io.BytesIO(image_bytes))
    
    # Check EXIF orientation
    try:
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(274, 1)  # 274 = Orientation tag
            # Rotate based on EXIF
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass  # No EXIF, proceed as-is
    
    return np.array(img)

def preprocess_for_model(image_bytes, target_size=300):
    """
    Full preprocessing pipeline.
    Input: Raw image bytes from phone upload
    Output: NumPy array [1, 300, 300, 3] ready for model.predict()
    """
    # Step 1: Fix rotation from phone camera
    img = auto_rotate(image_bytes)
    
    # Step 2: Convert RGB to BGR for OpenCV (if from PIL)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Step 3: Optional — Smart crop to center the object
    # This removes background clutter and focuses on the waste item
    img = smart_crop(img)
    
    # Step 4: Resize to model input size (300x300)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 5: Convert back to RGB (model was trained on RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 6: Add batch dimension and ensure float32
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    # ⚠️ IMPORTANT: Do NOT divide by 255
    # EfficientNetB3 has built-in preprocessing
    # img = img / 255.0  ← NEVER DO THIS
    
    return img

def smart_crop(img, padding=0.1):
    """
    Optional: Detect the main object and crop to it.
    Uses simple edge detection to find the waste item.
    Falls back to center crop if detection fails.
    """
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Fallback: center crop to square
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[top:top+min_dim, left:left+min_dim]
    
    # Find largest contour (assumed to be the object)
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    
    # Add padding around the object
    pad_x = int(cw * padding)
    pad_y = int(ch * padding)
    
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)
    
    cropped = img[y1:y2, x1:x2]
    
    # If crop is too small, fallback to center
    if cropped.shape[0] < 50 or cropped.shape[1] < 50:
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[top:top+min_dim, left:left+min_dim]
    
    return cropped

# ========== TEST ==========
if __name__ == "__main__":
    import sys
    import tensorflow as tf
    
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py photo.jpg")
        sys.exit(1)
    
    # Load image as bytes (simulating phone upload)
    with open(sys.argv[1], "rb") as f:
        image_bytes = f.read()
    
    # Preprocess
    processed = preprocess_for_model(image_bytes)
    
    print(f"Output shape: {processed.shape}")      # Should be (1, 300, 300, 3)
    print(f"Data type: {processed.dtype}")         # Should be float32
    print(f"Min pixel: {processed.min()}")         # Should be ~0
    print(f"Max pixel: {processed.max()}")         # Should be ~255
    print(f"Mean pixel: {processed.mean():.1f}")   # Should be ~100-150
    
    # Test prediction
    model = tf.keras.models.load_model("best_model_final.keras")
    preds = model.predict(processed, verbose=0)
    print(f"Prediction: {preds}")