import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────
# Label convention (consistent everywhere):
#   1 = bleeding   (positive class)
#   0 = non-bleeding
# ─────────────────────────────────────────────────────────────
LABEL_BLEED    = 1   # FIX 1: was 0 — bleeding must be the positive class (1)
LABEL_NONBLEED = 0   # FIX 1: was 1

# Default data root resolved relative to this file's location
# (utils/data_loaders.py → project root → TrainingDataset/)
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_DEFAULT_DATA = os.path.join(_PROJECT_ROOT, "TrainingDataset")


# =====================================================
# FILENAME UTILITY
# =====================================================

def _img_fname_to_ann_fname(img_fname):
    """
    Convert an image filename to its paired annotation filename.

    Dataset naming convention (with literal space before parenthesis):
        img- (1).png  →  ann- (1).png
        img- (42).png →  ann- (42).png

    Strategy: split on the first '-', keep everything after it
    (i.e. the ' (N).ext' suffix), then prepend 'ann-'.
    This is robust to any index value and preserves the space.
    """
    # "img- (1).png"  →  split on '-', maxsplit=1  →  ['img', ' (1).png']
    parts = img_fname.split("-", 1)
    if len(parts) == 2:
        return "ann-" + parts[1]          # → "ann- (1).png"
    # fallback: should never happen with this dataset
    return img_fname


# =====================================================
# XML READER
# =====================================================

def read_xml(path):
    """
    Parse a Pascal VOC XML annotation file.

    Returns:
        boxes   — list of normalised [x1, y1, x2, y2] boxes
        classes — list of int class labels (1=bleeding, 0=non-bleeding)

    Normalisation uses the <size> block inside the XML itself,
    so no external image dimensions are needed.
    """
    if not os.path.isfile(path):
        return [], []

    try:
        root  = ET.parse(path).getroot()
        size  = root.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)

        boxes, classes = [], []
        for obj in root.findall("object"):
            name_node = obj.find("name") or obj.find("n")
            name      = (name_node.text or "").strip().lower() \
                        if name_node is not None else ""
            cls       = LABEL_BLEED if "bleed" in name else LABEL_NONBLEED

            bb   = obj.find("bndbox")
            xmin = max(0.0, min(1.0, float(bb.find("xmin").text) / img_w))
            ymin = max(0.0, min(1.0, float(bb.find("ymin").text) / img_h))
            xmax = max(0.0, min(1.0, float(bb.find("xmax").text) / img_w))
            ymax = max(0.0, min(1.0, float(bb.find("ymax").text) / img_h))

            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(cls)

        return boxes, classes

    except Exception:
        return [], []


# =====================================================
# BOX UTILITIES
# =====================================================

def _boxarea(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def biggest_box(boxes):
    """Return (found: bool, largest [x1,y1,x2,y2] box)."""
    if not boxes:
        return False, [0.0, 0.0, 0.0, 0.0]
    best = max(boxes, key=_boxarea)
    if _boxarea(best) <= 0:
        return False, [0.0, 0.0, 0.0, 0.0]
    return True, best


# =====================================================
# CLASSIFICATION + BBOX LOADER
# =====================================================

def load_data(with_neg=True, aug=False, nums=1, data_root=None):
    """
    Load images, bounding boxes, and class labels.
    Annotation source: XML (Bounding boxes/XML/).

    Returns:
        images  — float32 (N, 224, 224, 3)  normalised [0, 1]
        boxes   — float32 (N, 4)             normalised [x1, y1, x2, y2]
        labels  — float32 (N,)               1=bleeding  0=non-bleeding
    """
    if data_root is None:
        data_root = _DEFAULT_DATA

    bleeding_path     = os.path.join(data_root, "bleeding")
    non_bleeding_path = os.path.join(data_root, "non-bleeding")
    xml_dir           = os.path.join(bleeding_path, "Bounding boxes", "XML")
    img_dir           = os.path.join(bleeding_path, "Images")
    IMG_EXTS          = (".png", ".jpg", ".jpeg", ".bmp")

    images, boxes, labels = [], [], []

    # ── BLEEDING ─────────────────────────────────────
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        img_bgr = cv2.imread(os.path.join(img_dir, fname))
        if img_bgr is None:
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        # FIX 2: derive annotation filename via prefix swap, not stem reuse.
        # "img- (1).png" → "ann- (1).xml"  (same index, correct prefix)
        ann_fname    = _img_fname_to_ann_fname(os.path.splitext(fname)[0]) + ".xml"
        xml_boxes, _ = read_xml(os.path.join(xml_dir, ann_fname))
        _, box       = biggest_box(xml_boxes)

        images.append(img)
        boxes.append(box)
        labels.append(float(LABEL_BLEED))

    # ── NON-BLEEDING ──────────────────────────────────
    if with_neg:
        nb_img_dir = os.path.join(non_bleeding_path, "images")
        for fname in sorted(os.listdir(nb_img_dir)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            img_bgr = cv2.imread(os.path.join(nb_img_dir, fname))
            if img_bgr is None:
                continue

            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

            images.append(img)
            boxes.append([0.0, 0.0, 0.0, 0.0])
            labels.append(float(LABEL_NONBLEED))

    return (
        np.array(images, dtype=np.float32),
        np.array(boxes,  dtype=np.float32),
        np.array(labels, dtype=np.float32),
    )


# =====================================================
# SEGMENTATION LOADER  (U-Net)
# =====================================================

def load_data_unet(aug=False, nums=1, data_root=None):
    """
    Load images and binary segmentation masks.
    Bleeding images only — non-bleeding images have no masks
    and must NOT be included here (all-zero masks would teach
    the U-Net to predict nothing, collapsing Dice and IoU to 0).

    Returns:
        images — float32 (N, 224, 224, 3)  normalised [0, 1]
        masks  — float32 (N, 224, 224)     values in {0.0, 1.0}
    """
    if data_root is None:
        data_root = _DEFAULT_DATA

    bleeding_path = os.path.join(data_root, "bleeding")
    img_dir       = os.path.join(bleeding_path, "Images")
    mask_dir      = os.path.join(bleeding_path, "Annotations")
    IMG_EXTS      = (".png", ".jpg", ".jpeg", ".bmp")

    images, masks = [], []

    # ── BLEEDING ONLY ────────────────────────────────
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        img_bgr = cv2.imread(os.path.join(img_dir, fname))
        if img_bgr is None:
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        # FIX 2 (segmentation): derive annotation filename via prefix swap.
        # "img- (1).png" → "ann- (1).png"
        ann_fname = _img_fname_to_ann_fname(fname)
        mask_path = os.path.join(mask_dir, ann_fname)

        # Fallback: try .bmp extension if .png not found
        if not os.path.exists(mask_path):
            ann_fname_bmp = _img_fname_to_ann_fname(
                os.path.splitext(fname)[0]
            ) + ".bmp"
            mask_path = os.path.join(mask_dir, ann_fname_bmp)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        images.append(img)
        masks.append(mask)

    # FIX 3: non-bleeding loop removed entirely.
    # Adding non-bleeding images with all-zero masks caused the U-Net
    # to collapse to predicting nothing (Dice=0, IoU=0).

    return (
        np.array(images, dtype=np.float32),
        np.array(masks,  dtype=np.float32),
    )