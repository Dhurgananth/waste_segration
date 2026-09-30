#!/usr/bin/env python3
"""
FIX AND BALANCE DATASET
=======================
Run this in your project folder.
Input:  merged_dataset_cleaned
Output: dataset_ready
"""
import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

SRC = "merged_dataset_cleaned"
DST = "dataset_ready"
TARGET_PER_CLASS = 1000   # Augment all classes to this count

# Step 1: Merge trash classes
TRASH_NAMES = {"miscellaneous_trash", "textile_trash", "trash"}

def merge_trash():
    os.makedirs(DST, exist_ok=True)
    
    for folder in os.listdir(SRC):
        src_path = os.path.join(SRC, folder)
        if not os.path.isdir(src_path) or folder.startswith("."):
            continue
        
        # Merge all trash variants into one folder
        if folder in TRASH_NAMES:
            dst_folder = "trash"
        else:
            dst_folder = folder
            
        dst_path = os.path.join(DST, dst_folder)
        os.makedirs(dst_path, exist_ok=True)
        
        for f in os.listdir(src_path):
            src_file = os.path.join(src_path, f)
            if not os.path.isfile(src_file):
                continue
            # Rename to avoid collisions
            new_name = f"{folder}_{f}"
            shutil.copy2(src_file, os.path.join(dst_path, new_name))
    
    print("Merged trash classes.")

# Step 2: Augment minority classes to TARGET_PER_CLASS
def augment_image(img):
    """Apply random augmentation."""
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    img = img.rotate(angle, expand=True)
    
    # Random flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    if random.random() > 0.5:
        img = ImageOps.flip(img)
    
    # Random brightness
    factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)
    
    # Random contrast
    factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    
    # Random zoom/crop
    w, h = img.size
    if random.random() > 0.5 and w > 300 and h > 300:
        crop_size = random.randint(int(min(w,h)*0.8), min(w,h))
        left = random.randint(0, w - crop_size)
        top = random.randint(0, h - crop_size)
        img = img.crop((left, top, left + crop_size, top + crop_size))
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    
    return img

def balance_classes():
    for class_name in sorted(os.listdir(DST)):
        class_path = os.path.join(DST, class_name)
        if not os.path.isdir(class_path):
            continue
            
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        current = len(images)
        
        print(f"\n{class_name}: {current} images")
        
        if current >= TARGET_PER_CLASS:
            # Randomly delete extras if way over
            if current > TARGET_PER_CLASS + 100:
                random.shuffle(images)
                for extra in images[TARGET_PER_CLASS:]:
                    os.remove(os.path.join(class_path, extra))
                print(f"   Trimmed to {TARGET_PER_CLASS}")
            else:
                print(f"   Kept as-is")
            continue
        
        # Augment up to target
        needed = TARGET_PER_CLASS - current
        print(f"   Augmenting +{needed}")
        
        while needed > 0:
            for img_name in images:
                if needed <= 0:
                    break
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        aug = augment_image(img)
                        new_name = f"aug_{needed:04d}_{img_name}"
                        aug.save(os.path.join(class_path, new_name), quality=95)
                        needed -= 1
                except Exception:
                    continue

def report():
    print("\n" + "=" * 50)
    print("FINAL DATASET READY")
    print("=" * 50)
    total = 0
    for cls in sorted(os.listdir(DST)):
        path = os.path.join(DST, cls)
        if not os.path.isdir(path):
            continue
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total += count
        print(f"{cls:15s}: {count}")
    print(f"{'TOTAL':15s}: {total}")
    print(f"\nOutput: {os.path.abspath(DST)}")

if __name__ == "__main__":
    # Clean old DST if exists
    if os.path.exists(DST):
        shutil.rmtree(DST)
    
    merge_trash()
    balance_classes()
    report()