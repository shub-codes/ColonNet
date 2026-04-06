"""
combo4_yolo_dataset.py  —  Combination 4
==========================================
Converts the WCEBleedGen dataset into YOLOv8-seg format.

YOLOv8-seg label format (one row per object):
    class_id  cx  cy  w  h  px1  py1  px2  py2  ...  pxN  pyN
where:
    class_id      : 0  (bleeding — only class)
    cx, cy, w, h  : YOLO bounding box in normalised coordinates
    px_i, py_i    : normalised polygon vertices of the segmentation mask

Dataset layout expected:
    data_root/
      bleeding/
        Images/              ← img- (N).png
        Bounding boxes/XML/  ← ann- (N).xml  (Pascal-VOC bounding boxes)
        Annotations/         ← ann- (N).png  (binary segmentation masks)

Output layout:
    yolo_root/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt   ← YOLO-seg format
      labels/val/*.txt
      dataset.yaml

Why YOLOv8n-seg vs plain YOLOv8n?
  YOLOv8n-seg predicts instance segmentation masks in addition to boxes
  using a prototype-based mask head.  This replaces UNet++ entirely.
  The extra overhead over YOLOv8n is ~15% FLOPs — negligible for our
  224×224 inputs.

Note on polygon extraction:
  The dataset supplies binary masks (Annotations/).  We convert each mask
  to a polygon via OpenCV's findContours and simplify with approxPolyDP.
  If no mask is found, the bounding box corners are used as a fallback
  polygon (a rectangle), which is still valid for YOLO-seg training.
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


IMG_SIZE = 224
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _img_fname_to_ann_fname(img_fname):
    """'img- (1).png' → 'ann- (1).png'"""
    parts = img_fname.split("-", 1)
    return ("ann-" + parts[1]) if len(parts) == 2 else img_fname


def _read_xml_box(xml_path):
    """
    Returns biggest bounding box from Pascal-VOC XML as
    YOLO (cx, cy, bw, bh) normalised to [0,1].
    Returns None on failure.
    """
    if not os.path.isfile(xml_path):
        return None
    try:
        root  = ET.parse(xml_path).getroot()
        size  = root.find("size")
        src_w = float(size.find("width").text)
        src_h = float(size.find("height").text)

        best, best_area = None, 0.0
        for obj in root.findall("object"):
            bb   = obj.find("bndbox")
            xmin = float(bb.find("xmin").text) / src_w
            ymin = float(bb.find("ymin").text) / src_h
            xmax = float(bb.find("xmax").text) / src_w
            ymax = float(bb.find("ymax").text) / src_h
            area = max(0, xmax - xmin) * max(0, ymax - ymin)
            if area > best_area:
                best_area = area
                best      = (xmin, ymin, xmax, ymax)

        if best is None or best_area <= 0:
            return None

        xmin, ymin, xmax, ymax = best
        return ((xmin + xmax) / 2,   # cx
                (ymin + ymax) / 2,   # cy
                xmax - xmin,         # bw
                ymax - ymin)         # bh
    except Exception:
        return None


def _mask_to_polygon(mask_path, epsilon_frac=0.002):
    """
    Loads a binary mask image and extracts the largest contour as a
    normalised polygon for YOLO-seg labels.

    epsilon_frac : polygon simplification tolerance as fraction of
                   arc-length.  0.002 gives smooth but compact polygons.

    Returns list of (x, y) normalised floats, or None if no contour found.
    """
    if not os.path.isfile(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                      interpolation=cv2.INTER_NEAREST)
    binary = (mask > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep largest contour only
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 4:
        return None

    # Simplify polygon
    eps     = epsilon_frac * cv2.arcLength(cnt, closed=True)
    approx  = cv2.approxPolyDP(cnt, eps, closed=True)
    pts     = approx.reshape(-1, 2)

    # Normalise to [0, 1]
    pts_norm = [(float(x) / IMG_SIZE, float(y) / IMG_SIZE) for x, y in pts]
    if len(pts_norm) < 3:
        return None
    return pts_norm


def _bbox_to_polygon(cx, cy, bw, bh):
    """Fallback: convert YOLO box to a 4-point rectangle polygon."""
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def prepare_yolo_seg_dataset(data_root, yolo_root, val_split=0.2, seed=42):
    """
    Prepares a YOLOv8-seg dataset from the bleeding subset of WCEBleedGen.

    Parameters
    ----------
    data_root : str   path to TrainingDataset/
    yolo_root : str   output directory
    val_split : float fraction used for validation
    seed      : int   random seed

    Returns
    -------
    yaml_path : str   path to dataset.yaml  (pass to YOLO.train)
    """
    bleeding_path = os.path.join(data_root, "bleeding")
    img_dir       = os.path.join(bleeding_path, "Images")
    xml_dir       = os.path.join(bleeding_path, "Bounding boxes", "XML")
    mask_dir      = os.path.join(bleeding_path, "Annotations")

    records = []   # (img_path, label_str, stem)

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue

        stem     = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, fname)

        # ── Bounding box ──────────────────────────────────────
        ann_stem = _img_fname_to_ann_fname(stem)
        xml_path = os.path.join(xml_dir, ann_stem + ".xml")
        box      = _read_xml_box(xml_path)
        if box is None:
            continue   # skip images with no valid box
        cx, cy, bw, bh = box

        # ── Segmentation polygon ──────────────────────────────
        polygon = None
        for ext in IMG_EXTS:
            mp = os.path.join(mask_dir, ann_stem + ext)
            if os.path.isfile(mp):
                polygon = _mask_to_polygon(mp)
                break

        if polygon is None:
            # Fallback: use bounding box as polygon
            polygon = _bbox_to_polygon(cx, cy, bw, bh)

        # ── Compose YOLO-seg label row ────────────────────────
        # Format: class_id cx cy bw bh  px1 py1 px2 py2 ...
        flat_poly = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
        label_str = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {flat_poly}"

        records.append((img_path, label_str, stem))

    if not records:
        raise RuntimeError(
            f"No valid (image, annotation) pairs found in {data_root}.\n"
            f"Check that XML files are under:\n  {xml_dir}\n"
            f"and are named 'ann- (N).xml'."
        )

    # ── Train / val split ─────────────────────────────────────
    np.random.seed(seed)
    idx      = np.random.permutation(len(records))
    n_val    = max(1, int(len(records) * val_split))
    val_set  = set(idx[:n_val].tolist())
    train_set = set(idx[n_val:].tolist())

    for split, indices in [("train", train_set), ("val", val_set)]:
        img_out = os.path.join(yolo_root, "images", split)
        lbl_out = os.path.join(yolo_root, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for i in indices:
            img_path, label_str, stem = records[i]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(img_out, stem + ".jpg"), img)

            with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
                f.write(label_str + "\n")

    # ── YAML ──────────────────────────────────────────────────
    yaml_path = os.path.join(yolo_root, "dataset.yaml")
    abs_root  = os.path.abspath(yolo_root)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_root}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['bleeding']\n")

    print(f"[YOLO-seg dataset] {len(train_set)} train / {len(val_set)} val")
    print(f"[YOLO-seg dataset] YAML → {yaml_path}")
    return yaml_path