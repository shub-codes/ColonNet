import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────
# Label convention:
#   1 = bleeding   (positive class)
#   0 = non-bleeding
# FIX 1: Original had LABEL_BLEED=0, LABEL_NONBLEED=1 which inverts
# the positive class. sigmoid + BinaryCrossentropy expects 1 = positive.
# ─────────────────────────────────────────────────────────────
LABEL_BLEED    = 1
LABEL_NONBLEED = 0

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_DEFAULT_DATA = r"C:\Users\Shubham\Desktop\ColonNet\TrainingDataset"


# =====================================================
# FILENAME UTILITY
# =====================================================

def _img_fname_to_ann_fname(img_fname):
    """
    Convert image filename to its paired annotation filename.
    Dataset convention (note literal space before parenthesis):
        img- (1).png  →  ann- (1).png
        img- (42).xml →  ann- (42).xml
    Splits on the first '-' and prepends 'ann-', preserving the suffix.
    """
    parts = img_fname.split("-", 1)
    if len(parts) == 2:
        return "ann-" + parts[1]
    return img_fname  # fallback


# =====================================================
# XML READER
# =====================================================

def read_xml(path):
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

            cls = LABEL_BLEED if "bleed" in name else LABEL_NONBLEED

            bb   = obj.find("bndbox")
            xmin = float(bb.find("xmin").text) / img_w
            ymin = float(bb.find("ymin").text) / img_h
            xmax = float(bb.find("xmax").text) / img_w
            ymax = float(bb.find("ymax").text) / img_h

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
    if not boxes:
        return False, [0.0, 0.0, 0.0, 0.0]

    best = max(boxes, key=_boxarea)
    if _boxarea(best) <= 0:
        return False, [0.0, 0.0, 0.0, 0.0]

    return True, best


# =====================================================
# MAIN LOADER
# =====================================================

def load_data(with_neg=True, aug=False, nums=1, data_root=None):
    """
    Returns:
        images  — float32 (N, 224, 224, 3)
        boxes   — float32 (N, 4)  normalised [x1,y1,x2,y2]
        labels  — float32 (N,)    1=bleeding  0=non-bleeding
    """
    if data_root is None:
        data_root = _DEFAULT_DATA

    bleeding_path     = os.path.join(data_root, "bleeding")
    non_bleeding_path = os.path.join(data_root, "non-bleeding")

    xml_dir  = os.path.join(bleeding_path, "Bounding boxes", "XML")
    img_dir  = os.path.join(bleeding_path, "Images")
    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    images, boxes, labels = [], [], []

    # ── BLEEDING ──
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue

        img_bgr = cv2.imread(os.path.join(img_dir, fname))
        if img_bgr is None:
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        # FIX 2: derive annotation filename by prefix swap, not stem reuse.
        # "img- (1).png" → stem "img- (1)" → ann stem "ann- (1)" → "ann- (1).xml"
        # Original used stem + ".xml" which looked for "img- (1).xml" — never found.
        ann_stem     = _img_fname_to_ann_fname(os.path.splitext(fname)[0])
        xml_boxes, _ = read_xml(os.path.join(xml_dir, ann_stem + ".xml"))
        _, box       = biggest_box(xml_boxes)

        images.append(img)
        boxes.append(box)
        labels.append(float(LABEL_BLEED))   # FIX 1: directly use constant (=1.0)

    # ── NON-BLEEDING ──
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
            labels.append(float(LABEL_NONBLEED))   # FIX 1: directly use constant (=0.0)

    # FIX 3: valid_mask array removed from return value.
    # It was computed but then commented out of the return tuple, making
    # the return a 3-tuple. Any caller unpacking 4 values would crash.
    # The combined_box_loss in training.py derives its own valid mask
    # from box area, so this array is not needed here.
    return (
        np.array(images, dtype=np.float32),
        np.array(boxes,  dtype=np.float32),
        np.array(labels, dtype=np.float32),
    )


# =====================================================
# SEGMENTATION LOADER
# =====================================================

def load_data_unet(aug=False, nums=1, data_root=None, include_neg=False):
    """
    Loads bleeding images and their segmentation masks.
    include_neg=False by default — non-bleeding images carry no mask
    signal and collapse the U-Net to predicting all-zeros if included.
    """
    if data_root is None:
        data_root = _DEFAULT_DATA

    bleeding_path     = os.path.join(data_root, "bleeding")
    non_bleeding_path = os.path.join(data_root, "non-bleeding")

    img_dir  = os.path.join(bleeding_path, "Images")
    mask_dir = os.path.join(bleeding_path, "Annotations")
    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    images, masks = [], []

    # ── BLEEDING ──
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue

        img_bgr = cv2.imread(os.path.join(img_dir, fname))
        if img_bgr is None:
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        # FIX 2: derive annotation filename by prefix swap.
        # "img- (1).png" → "ann- (1).png"  (not "img- (1).png")
        ann_fname = _img_fname_to_ann_fname(fname)
        mask_path = os.path.join(mask_dir, ann_fname)

        if not os.path.exists(mask_path):
            # fallback: try .bmp extension
            ann_bmp   = _img_fname_to_ann_fname(os.path.splitext(fname)[0]) + ".bmp"
            mask_path = os.path.join(mask_dir, ann_bmp)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        images.append(img)
        masks.append(mask)

    # ── OPTIONAL NON-BLEEDING ──
    # include_neg defaults to False. Only set True if you have real masks
    # for non-bleeding frames. All-zero masks teach the U-Net to predict
    # nothing and will collapse Dice and IoU to zero.
    if include_neg:
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
            masks.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))

    return (
        np.array(images, dtype=np.float32),
        np.array(masks,  dtype=np.float32),
    )