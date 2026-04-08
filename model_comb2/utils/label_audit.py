"""
fix_labels.py
─────────────
Moves degenerate bounding-box samples (box area > 0.90) to a backup folder.
For each bad sample index N it moves:

  TrainingDataset/bleeding/Images/                  img- (N).png
  TrainingDataset/bleeding/Annotations/             ann- (N).png
  TrainingDataset/bleeding/Bounding boxes/TXT/      img- (N).txt
  TrainingDataset/bleeding/Bounding boxes/XML/      img- (N).xml
  TrainingDataset/bleeding/Bounding boxes/YOLO_TXT/ img- (N).txt

All moved files go to:
  TrainingDataset/bleeding/_degenerate_backup/

Run ONCE before training:
    python pythonFiles/fix_labels.py
"""

import os
import sys
import re
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIG — adjust DATASET_ROOT if needed
# ─────────────────────────────────────────────────────────────
DATASET_ROOT = r'C:\Users\Shubham\Desktop\ColonNet\TrainingDataset\bleeding'

DIR_IMAGES = os.path.join(DATASET_ROOT, "Images")
DIR_ANN    = os.path.join(DATASET_ROOT, "Annotations")
DIR_TXT    = os.path.join(DATASET_ROOT, "Bounding boxes", "TXT")
DIR_XML    = os.path.join(DATASET_ROOT, "Bounding boxes", "XML")
DIR_YOLO   = os.path.join(DATASET_ROOT, "Bounding boxes", "YOLO_TXT")

BACKUP_DIR = os.path.join(DATASET_ROOT, "_degenerate_backup")

AREA_THRESHOLD = 0.90


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def extract_index(filename):
    m = re.search(r'\(\s*(\d+)\s*\)', filename)
    return int(m.group(1)) if m else None


def read_box_plain(txt_path):
    if not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path) as f:
            line = f.readline().strip()
        if not line:
            return None
        floats = []
        for t in line.split():
            try:
                floats.append(float(t))
            except ValueError:
                pass
        if len(floats) >= 4:
            return np.array(floats[-4:], dtype=np.float32)
    except Exception:
        pass
    return None


def read_box_yolo(yolo_path):
    if not os.path.exists(yolo_path):
        return None
    try:
        with open(yolo_path) as f:
            line = f.readline().strip()
        tokens = line.split()
        if len(tokens) < 5:
            return None
        cx, cy, w, h = float(tokens[1]), float(tokens[2]), \
                       float(tokens[3]), float(tokens[4])
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                        dtype=np.float32)
    except Exception:
        pass
    return None


def move_to_backup(src_path):
    """Move src_path into BACKUP_DIR. Skips silently if file doesn't exist."""
    if not os.path.exists(src_path):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dest = os.path.join(BACKUP_DIR, os.path.basename(src_path))
    shutil.move(src_path, dest)
    print(f"  backed up: {os.path.basename(src_path)}")


# ─────────────────────────────────────────────────────────────
# Step 1 — Discover all image indices
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("fix_labels.py — degenerate sample removal")
print("=" * 55)

if not os.path.isdir(DIR_IMAGES):
    print(f"\n✗ Images folder not found:\n    {DIR_IMAGES}")
    print("  Update DATASET_ROOT at the top of this script.")
    sys.exit(1)

index_to_imgfile = {}
for fname in os.listdir(DIR_IMAGES):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        idx = extract_index(fname)
        if idx is not None:
            index_to_imgfile[idx] = fname

print(f"\nTotal images found : {len(index_to_imgfile)}")


# ─────────────────────────────────────────────────────────────
# Step 2 — Identify degenerate indices
# ─────────────────────────────────────────────────────────────
print("Reading bounding boxes …")

degenerate_indices = []

for idx in sorted(index_to_imgfile.keys()):
    stem = f"img- ({idx})"

    box = read_box_plain(os.path.join(DIR_TXT, stem + ".txt"))
    if box is None:
        box = read_box_yolo(os.path.join(DIR_YOLO, stem + ".txt"))
    if box is None:
        continue

    area = (box[2] - box[0]) * (box[3] - box[1])
    if area > AREA_THRESHOLD:
        degenerate_indices.append(idx)

print(f"Degenerate samples : {len(degenerate_indices)}")
print(f"Indices            : {degenerate_indices}\n")

if not degenerate_indices:
    print("✓ No degenerate samples — dataset is already clean.")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────
# Step 3 — Move only the degenerate files to backup
# ─────────────────────────────────────────────────────────────
print(f"Moving degenerate files to:\n  {BACKUP_DIR}\n")

for idx in degenerate_indices:
    img_stem = f"img- ({idx})"
    ann_stem = f"ann- ({idx})"
    print(f"[{idx}]")
    move_to_backup(os.path.join(DIR_IMAGES, index_to_imgfile[idx]))
    move_to_backup(os.path.join(DIR_ANN,    ann_stem + ".png"))
    move_to_backup(os.path.join(DIR_TXT,    img_stem + ".txt"))
    move_to_backup(os.path.join(DIR_XML,    img_stem + ".xml"))
    move_to_backup(os.path.join(DIR_YOLO,   img_stem + ".txt"))


# ─────────────────────────────────────────────────────────────
# Step 4 — Final count check
# ─────────────────────────────────────────────────────────────
remaining = [
    f for f in os.listdir(DIR_IMAGES)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
expected = len(index_to_imgfile) - len(degenerate_indices)
print(f"\nImages remaining : {len(remaining)}  (expected {expected})")

if len(remaining) == expected:
    print("\n✓ Done. Next steps:")
    print("  1. Delete  SavedModels/rs_best_params.json")
    print("  2. Delete  SavedModels/CheckPoint1.keras")
    print("  3. Delete  SavedModels/classNbox.keras")
    print("  4. Run     python pythonFiles/training.py")
else:
    print("\n⚠  Count mismatch — check manually.")