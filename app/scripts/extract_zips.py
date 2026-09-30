# import os
# import sys
# import shutil
# import zipfile
# import hashlib
# import argparse
# from pathlib import Path
# from collections import defaultdict,Counter

# IMAGE_EXISTS = {".jpg",".jpeg",".png",".gif",".webp",".tiff"}

# CLASS_MAP = {}

# def print_header(text):
#     print("\n" + "=" * 60)

#     print(f" {text}")
#     print("==" * 60)

# def extract_zip(zip_path,extract_to):
#     if not os.path.exists(zip_path):print(f"file not found {zip_path}")
#     sys.exit(1)

#     zip_name = Path(zip_path).stem
#     extract_path = os.path.join(extract_to,zip_name)
#     os.makerdirs(extract_path, exist_ok = True)
#     print(f" Extracting:{zip_path}")
#     print(f"size: {os.path.getsize(zip_path)/ 1024 /1024:.1f} MB")

#     with zipfile.ZipFile(zip_path,'r') as zf:

#         for member in zf.namelist():
#             if member.startswith('/') or '..' in member:
#                 print(f" Skipping suspicious path:{member}")
#                 continue
#             zf.extractall(extract_path)

#         print(f"Extracted to : {extract_path}")
#         return extract_path

# def scan_folder(path,depth=0,max_depth = 3):
#     images = []
#     folders = []

#     for root, dirs,files in os.walk(path):
#         current_depth = root.replace(path,'').count(os.sep)
#         if current_depth > max_depth:
#             del dirs[:]
#             continue

#         img_files = [f for f in files if Path(f).sufflix.lower() in IMAGE_EXISTS]

#         if img_files:
#            rel = os.path.relpath(root,path)
#            folders.append({
#                'path' : root,
#                'rel' : rel,
#                'name':os.path.basename(root),
#                'count' : len(img_files),
#                'images':img_files
#              })
#            images.extend([os.path.join(root,f) for f in img_files])
#         if img_files and not any(
#             any(Path(f).suffix.lower() in IMAGE_EXISTS for f in os.listdir(os.path.join(root,d)))

#             for d in dirs if os.path.isdir(os.path.join(root,d))
#          ):
#             dirs[:] =[]
#         return images,folders

#     def show_details(name,extract_path):

#         print_header(f"Details : {name}")

#         images,folders = scan_folder(extract_path)

#         print(f"/n Total folders with images:{len(folders)}")

#         print(f"Total images found:len{len(images)}")

#         if not folders:
#             print(" NO images folders found")
#             return []

#         print(f"\n{'Folder Name': <25} {'images':> 8} {'Depth':>6}{'Sample files'}")
#         print("-" * 70)

#         for f in folders:
#             depth =f['rel']
             