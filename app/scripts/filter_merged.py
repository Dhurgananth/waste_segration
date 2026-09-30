#!/usr/bin/env python3
# """
# FILTER MERGED DATASET
# =====================
# Run this AFTER extract_merge.py and BEFORE training.
# """
import os
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageStat
import hashlib

SRC = "merged_dataset"
DST = "merged_dataset_cleaned"
MIN_SIZE = 128          # Reject tiny images
MAX_SIZE = 4096         # Reject absurdly huge images
MIN_DIM = 224           # Must be at least 224x224 for EfficientNet
BLUR_THRESHOLD = 100    # Laplacian variance below this = blurry
DARK_THRESHOLD = 20     # Mean brightness below this = too dark

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def get_image_hash(img):
    """Hash of resized image content to catch near-duplicates."""
    small = img.resize((64, 64), Image.Resampling.LANCZOS).convert("L")
    return hashlib.md5(small.tobytes()).hexdigest()

def is_blurry(img):
    """Check if image is blurry using Laplacian variance approximation."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_too_dark(img):
    """Check mean brightness."""
    stat = ImageStat.Stat(img.convert("L"))
    return stat.mean[0]

def filter_dataset():
    os.makedirs(DST, exist_ok=True)
    
    stats = {
        "total_scanned": 0,
        "kept": 0,
        "corrupted": 0,
        "too_small": 0,
        "too_big": 0,
        "wrong_ext": 0,
        "blurry": 0,
        "too_dark": 0,
        "duplicates": 0,
        "converted": 0,
    }
    
    seen_hashes = set()
    class_counts = Counter()
    rejected_log = defaultdict(list)
    
    # Walk through each class folder
    for class_name in sorted(os.listdir(SRC)):
        class_path = os.path.join(SRC, class_name)
        if not os.path.isdir(class_path) or class_name.startswith("."):
            continue
            
        target_class = os.path.join(DST, class_name)
        os.makedirs(target_class, exist_ok=True)
        
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            ext = Path(filename).suffix.lower()
            stats["total_scanned"] += 1
            
            # 1. Wrong extension
            if ext not in VALID_EXTS:
                stats["wrong_ext"] += 1
                rejected_log[class_name].append(("wrong_ext", filename))
                continue
            
            # 2. Try open with PIL
            try:
                with Image.open(filepath) as img:
                    # 3. Check size
                    w, h = img.size
                    if w < MIN_DIM or h < MIN_DIM:
                        stats["too_small"] += 1
                        rejected_log[class_name].append(("too_small", filename))
                        continue
                    if w > MAX_SIZE or h > MAX_SIZE:
                        stats["too_big"] += 1
                        continue
                    
                    # 4. Convert to RGB if needed
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                        stats["converted"] += 1
                    
                    # 5. Check for near-duplicates
                    img_hash = get_image_hash(img)
                    if img_hash in seen_hashes:
                        stats["duplicates"] += 1
                        rejected_log[class_name].append(("duplicate", filename))
                        continue
                    seen_hashes.add(img_hash)
                    
                    # 6. Check blur (optional, requires opencv)
                    try:
                        blur_score = is_blurry(img)
                        if blur_score < BLUR_THRESHOLD:
                            stats["blurry"] += 1
                            rejected_log[class_name].append(("blurry", filename))
                            continue
                    except ImportError:
                        pass  # Skip blur check if cv2 not installed
                    
                    # 7. Check darkness
                    brightness = is_too_dark(img)
                    if brightness < DARK_THRESHOLD:
                        stats["too_dark"] += 1
                        rejected_log[class_name].append(("dark", filename))
                        continue
                    
                    # 8. Save clean copy as JPG
                    new_name = Path(filename).stem + ".jpg"
                    dest_path = os.path.join(target_class, new_name)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(target_class, f"{Path(filename).stem}_{counter:03d}.jpg")
                        counter += 1
                    
                    img.save(dest_path, quality=95, optimize=True)
                    stats["kept"] += 1
                    class_counts[class_name] += 1
                    
            except Exception as e:
                stats["corrupted"] += 1
                rejected_log[class_name].append(("corrupted", filename))
    
    # Print report
    print("=" * 60)
    print("FILTER REPORT")
    print("=" * 60)
    print(f"Total scanned:    {stats['total_scanned']}")
    print(f"Kept:             {stats['kept']} ({100*stats['kept']/max(stats['total_scanned'],1):.1f}%)")
    print(f"Corrupted:        {stats['corrupted']}")
    print(f"Too small:        {stats['too_small']}")
    print(f"Too big:          {stats['too_big']}")
    print(f"Wrong extension:  {stats['wrong_ext']}")
    print(f"Near-duplicates:  {stats['duplicates']}")
    print(f"Too blurry:       {stats['blurry']}")
    print(f"Too dark:         {stats['too_dark']}")
    print(f"Converted to RGB: {stats['converted']}")
    
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION AFTER FILTERING")
    print("=" * 60)
    total = sum(class_counts.values())
    for cls, count in class_counts.most_common():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"{cls:15s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Flag imbalance
    if class_counts:
        max_c = max(class_counts.values())
        min_c = min(class_counts.values())
        ratio = max_c / min_c
        print(f"\nImbalance ratio (max/min): {ratio:.1f}:1")
        if ratio > 3:
            print("⚠️  WARNING: Heavy imbalance detected. Augment minority classes!")
        elif ratio > 1.5:
            print("⚠️  Moderate imbalance. Consider class weights.")
        else:
            print("✅ Good balance.")
    
    # Show sample rejects per class
    print("\n" + "=" * 60)
    print("SAMPLE REJECTIONS PER CLASS")
    print("=" * 60)
    for cls in sorted(rejected_log.keys())[:5]:
        items = rejected_log[cls][:3]
        print(f"\n{cls}:")
        for reason, fname in items:
            print(f"   [{reason}] {fname}")
    
    print(f"\n✅ Clean dataset saved to: {os.path.abspath(DST)}")

if __name__ == "__main__":
    filter_dataset()