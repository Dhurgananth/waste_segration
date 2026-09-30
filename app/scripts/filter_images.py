#!/usr/bin/env python3
"""
IMAGE FILTERING SCRIPT
======================

Removes bad images BEFORE training. This is critical for real-world accuracy.
Bad training images = bad model predictions.

WHAT IT REMOVES:
- Corrupted files (can't be opened)
- Too small (< 100x100 pixels)
- Too dark or too bright (blank photos)
- Too blurry (Laplacian variance < 50)
- Wrong aspect ratio (extremely tall/wide)

USAGE:
    python filter_images.py --input "merged_dataset" --output "cleaned_dataset"
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import argparse

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# Quality thresholds
MIN_SIZE = 100          # Minimum width or height
MAX_SIZE = 4096         # Maximum width or height
MIN_FILESIZE = 1024     # 1KB (remove empty files)
MAX_FILESIZE = 10 * 1024 * 1024  # 10MB
MIN_BRIGHTNESS = 15     # Too dark
MAX_BRIGHTNESS = 250    # Too bright
MIN_SHARPNESS = 50      # Laplacian variance (blur detection)
MAX_ASPECT_RATIO = 3.0  # Reject extremely tall/wide images


def check_image(path):
    """
    Check if an image is valid and good quality.
    Returns: (is_valid: bool, reason: str)
    """
    # Check file size
    fsize = os.path.getsize(path)
    if fsize < MIN_FILESIZE:
        return False, "Too small (< 1KB)"
    if fsize > MAX_FILESIZE:
        return False, "Too large (> 10MB)"

    # Try to open with OpenCV
    img = cv2.imread(path)
    if img is None:
        return False, "Cannot read (corrupted)"

    h, w = img.shape[:2]

    # Check dimensions
    if w < MIN_SIZE or h < MIN_SIZE:
        return False, f"Too small ({w}x{h})"
    if w > MAX_SIZE or h > MAX_SIZE:
        return False, f"Too large ({w}x{h})"

    # Check aspect ratio
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT_RATIO:
        return False, f"Bad aspect ratio ({aspect:.1f}:1)"

    # Check brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < MIN_BRIGHTNESS:
        return False, f"Too dark (brightness: {mean_brightness:.0f})"
    if mean_brightness > MAX_BRIGHTNESS:
        return False, f"Too bright (brightness: {mean_brightness:.0f})"

    # Check blur (Laplacian variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < MIN_SHARPNESS:
        return False, f"Too blurry (sharpness: {lap_var:.1f})"

    return True, "OK"


def filter_dataset(input_dir, output_dir):
    """
    Filter all images in input_dir and copy good ones to output_dir.
    """
    print("=" * 65)
    print("  IMAGE FILTERING")
    print("=" * 65)
    print(f"\n  Input:  {os.path.abspath(input_dir)}")
    print(f"  Output: {os.path.abspath(output_dir)}")

    os.makedirs(output_dir, exist_ok=True)

    total_original = 0
    total_kept = 0
    total_removed = 0
    removal_reasons = Counter()

    for class_name in sorted(os.listdir(input_dir)):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        target_path = os.path.join(output_dir, class_name)
        os.makedirs(target_path, exist_ok=True)

        kept = 0
        removed = 0

        for filename in os.listdir(class_path):
            if Path(filename).suffix.lower() not in IMAGE_EXTS:
                continue

            src = os.path.join(class_path, filename)
            total_original += 1

            valid, reason = check_image(src)

            if valid:
                shutil.copy2(src, os.path.join(target_path, filename))
                kept += 1
            else:
                removed += 1
                removal_reasons[reason] += 1

        total_kept += kept
        total_removed += removed
        print(f"  {class_name:<15}  kept: {kept:>4}  removed: {removed:>4}")

    print("\n" + "=" * 65)
    print("  FILTERING COMPLETE")
    print("=" * 65)
    print(f"\n  Original images: {total_original}")
    print(f"  Kept:            {total_kept}")
    print(f"  Removed:         {total_removed}")
    print(f"  Keep rate:       {total_kept/max(total_original,1)*100:.1f}%")

    if removal_reasons:
        print("\n  Removal reasons:")
        for reason, count in removal_reasons.most_common():
            print(f"     {reason:<30} {count:>4}")

    print(f"\n  Clean dataset saved to: {os.path.abspath(output_dir)}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Filter bad images before training')
    parser.add_argument('--input', default='merged_dataset', help='Input folder')
    parser.add_argument('--output', default='cleaned_dataset', help='Output folder')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input folder not found: {args.input}")
        print("Run extract_merge.py first to create the merged dataset.")
        return

    filter_dataset(args.input, args.output)

    print("\nNext step:")
    print(f"   python train_best_model.py --data \"{args.output}\"")


if __name__ == "__main__":
    main()