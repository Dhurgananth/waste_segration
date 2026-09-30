# #!/usr/bin/env python3
# """
# WASTE DATASET: EXTRACT + ANALYZE + MERGE
# =========================================

# For Windows with your exact file paths:
#     C:\Users\Dhurgannath\Downloads\dataset-resized.zip    (TrashNet)
#     C:\Users\Dhurgannath\Downloads\realwaste.zip          (RealWaste UCI)

# HOW TO RUN:
# -----------
# 1. Open VS Code terminal (Ctrl + `)
# 2. Make sure this script is in your project folder
# 3. Run: python extract_merge.py

# The script will:
# - Extract both zips
# - Show you exactly what is inside each one
# - Merge them into one clean folder
# - Remove duplicate images
# - Show final class counts
# """

import os
import sys
import shutil
import zipfile
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

# ========== YOUR FILE PATHS ==========
# Change these if your files are in a different location
ZIP_FILE_1 = r"C:\Users\Dhurgannath\Downloads\dataset-resized.zip"
ZIP_FILE_2 = r"C:\Users\Dhurgannath\Downloads\realwaste.zip"

# Where to extract and merge
EXTRACT_FOLDER = "extracted"
MERGED_FOLDER = "merged_dataset"

# ========== CLASS MAPPING ==========
# TrashNet has 6 classes. RealWaste has 9 classes.
# We map RealWaste classes to match TrashNet + add organic for food/vegetation.

CLASS_MAP = {
    'cardboard': 'cardboard',
    'glass': 'glass',
    'metal': 'metal',
    'paper': 'paper',
    'plastic': 'plastic',
    'trash': 'trash',
    'food_organics': 'organic',
    'food organics': 'organic',
    'miscellaneous': 'trash',
    'misc': 'trash',
    'textiles': 'trash',
    'vegetation': 'organic',
    'vegetation_': 'organic',
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}


def print_box(title):
    print("\n" + "=" * 65)
    print("  " + title)
    print("=" * 65)


def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print("\nERROR: FILE NOT FOUND: " + zip_path)
        print("Please check the path and make sure the file exists.")
        return None

    zip_name = Path(zip_path).stem
    out_path = os.path.join(extract_to, zip_name)
    os.makedirs(out_path, exist_ok=True)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print("\nExtracting: " + os.path.basename(zip_path))
    print("   Full path: " + zip_path)
    print("   File size: " + str(round(size_mb, 1)) + " MB")
    print("   Output to: " + out_path)

    try:    
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_path)
        print("   Done!")
        return out_path
    except zipfile.BadZipFile:
        print("   Error: File is not a valid zip file!")
        return None


def scan_images(folder_path):
    results = []
    total_images = 0

    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        images = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
        if images:
            rel = os.path.relpath(root, folder_path)
            results.append({
                'path': root,
                'rel': rel,
                'name': os.path.basename(root),
                'count': len(images),
                'images': images[:3]
            })
            total_images += len(images)

    return results, total_images


def show_details(name, extract_path):
    print_box("INSIDE: " + name)

    folders, total = scan_images(extract_path)

    if not folders:
        print("\nNo image folders found!")
        return []

    print("\nFolders with images: " + str(len(folders)))
    print("Total images: " + str(total) + "\n")

    print("Folder Name".ljust(22) + "Images".rjust(8) + "   Sample Files")
    print("-" * 65)
    for f in folders:
        samples = ", ".join(f["images"])
        if f["count"] > 3:
            samples += " ... (+" + str(f["count"]-3) + ")"
        print(f["name"].ljust(22) + str(f["count"]).rjust(8) + "   " + samples)

    print("\nFolder Tree:")
    for root, dirs, files in os.walk(extract_path):
        level = root.replace(extract_path, "").count(os.sep)
        if level > 2:
            continue
        indent = "  " * level
        folder_name = os.path.basename(root) or os.path.basename(extract_path)
        imgs = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
        if imgs or dirs:
            line = indent + "[FOLDER] " + folder_name + "/"
            if imgs:
                line += "  (" + str(len(imgs)) + " images)"
            print(line)

    return folders


def md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def merge(extracted_paths):
    print_box("MERGING DATASETS")

    os.makedirs(MERGED_FOLDER, exist_ok=True)
    seen_hashes = set()
    dup_count = 0
    stats = Counter()

    all_classes = defaultdict(list)

    for src_path in extracted_paths:
        folders, _ = scan_images(src_path)
        for folder in folders:
            raw_name = folder["name"].lower().strip().replace(" ", "_").replace("-", "_")
            unified = CLASS_MAP.get(raw_name, raw_name)
            all_classes[unified].append(folder["path"])

    print("\nUnified classes found: " + str(len(all_classes)))
    for cls, sources in sorted(all_classes.items()):
        img_count = sum(len([f for f in os.listdir(p) if Path(f).suffix.lower() in IMAGE_EXTS]) for p in sources)
        print("   " + cls.ljust(15) + " <- " + str(len(sources)) + " folder(s), ~" + str(img_count) + " images")

    print("\nCopying images (removing duplicates)...")

    for unified_class, source_folders in sorted(all_classes.items()):
        target_dir = os.path.join(MERGED_FOLDER, unified_class)
        os.makedirs(target_dir, exist_ok=True)

        for src_dir in source_folders:
            for filename in os.listdir(src_dir):
                if Path(filename).suffix.lower() not in IMAGE_EXTS:
                    continue

                src_file = os.path.join(src_dir, filename)
                h = md5(src_file)
                if h in seen_hashes:
                    dup_count += 1
                    continue
                seen_hashes.add(h)

                dest_file = os.path.join(target_dir, filename)
                counter = 1
                name, ext = os.path.splitext(filename)
                while os.path.exists(dest_file):
                    dest_file = os.path.join(target_dir, name + "_" + str(counter).zfill(3) + ext)
                    counter += 1

                shutil.copy2(src_file, dest_file)
                stats[unified_class] += 1

    print_box("MERGE COMPLETE")
    print("\nOutput folder: " + os.path.abspath(MERGED_FOLDER) + "\n")

    print("Class".ljust(15) + "Images".rjust(8))
    print("-" * 30)
    for cls in sorted(stats.keys()):
        print(cls.ljust(15) + str(stats[cls]).rjust(8))
    print("-" * 30)
    print("TOTAL".ljust(15) + str(sum(stats.values())).rjust(8))
    print("\nDuplicates removed: " + str(dup_count))

    return MERGED_FOLDER


def main():
    print_box("WASTE DATASET TOOL")
    print("\nThis script will:")
    print("  1. Extract your two zip files")
    print("  2. Show you what is inside each one")
    print("  3. Merge them into one clean folder")
    print("  4. Remove duplicate images")

    print("\nChecking files...")
    for zf in [ZIP_FILE_1, ZIP_FILE_2]:
        if os.path.exists(zf):
            size = os.path.getsize(zf) / (1024*1024)
            print("   [OK] Found: " + os.path.basename(zf) + " (" + str(round(size,1)) + " MB)")
        else:
            print("   [MISSING] " + zf)
            print("\nPlease edit ZIP_FILE_1 and ZIP_FILE_2 at the top of this script")
            print("to match your actual file paths.")
            return

    os.makedirs(EXTRACT_FOLDER, exist_ok=True)

    path1 = extract_zip(ZIP_FILE_1, EXTRACT_FOLDER)
    path2 = extract_zip(ZIP_FILE_2, EXTRACT_FOLDER)

    if not path1 or not path2:
        print("\nExtraction failed. Please check your zip files.")
        return

    show_details("TrashNet (dataset-resized)", path1)
    show_details("RealWaste", path2)

    merged = merge([path1, path2])

    print_box("NEXT STEPS")
    print("\nYour merged dataset is ready at:")
    print("   " + os.path.abspath(merged))
    print("\nNow run these scripts in order:")
    print("   1. python filter_images.py     (remove bad/blurry images)")
    print("   2. python split_data.py        (split into train/val/test)")
    print("   3. python train_model.py       (train EfficientNetB3)")


if __name__ == "__main__":
    main()