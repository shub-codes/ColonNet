import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import albumentations as alb

IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────
# Label convention (consistent everywhere):
#   1 = bleeding   (positive class)
#   0 = non-bleeding
# ─────────────────────────────────────────────────────────────
LABEL_BLEED    = 1
LABEL_NONBLEED = 0

# Default data root resolved relative to this file's location
# (utils/data_loaders.py → project root → TrainingDataset/)
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_DEFAULT_DATA = r"C:\Users\Shubham\Desktop\ColonNet\TrainingDataset"

# =====================================================
# FILENAME UTILITY
# =====================================================

def _img_fname_to_ann_fname(img_fname):
    """
    Convert an image filename to its paired MASK annotation filename.
    Use ONLY for segmentation masks (Annotations/ folder).

    Dataset naming convention (with literal space before parenthesis):
        img- (1).png  →  ann- (1).png
        img- (42).png →  ann- (42).png

    NOTE: Do NOT use this for XML bounding-box files.
          XML files keep the img- prefix: img- (1).xml
    """
    parts = img_fname.split("-", 1)
    if len(parts) == 2:
        return "ann-" + parts[1]   # → "ann- (1).png"
    return img_fname               # fallback — should never happen


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

    aug=True multiplies the dataset via albumentations — essential
    for small datasets to prevent overfitting.

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

    if aug:
        augmentor = alb.Compose([
            alb.BBoxSafeRandomCrop(),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(),
            alb.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        ], bbox_params=alb.BboxParams(format='albumentations',
                                      label_fields=['class_labels']))

    # ── BLEEDING ─────────────────────────────────────
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        img_bgr = cv2.imread(os.path.join(img_dir, fname))
        if img_bgr is None:
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        # FIX: XML files keep the img- prefix — use stem directly.
        # "img- (1).png" → stem = "img- (1)" → "img- (1).xml"
        # Do NOT call _img_fname_to_ann_fname here — that produces ann- (1).xml
        stem         = os.path.splitext(fname)[0]
        xml_boxes, _ = read_xml(os.path.join(xml_dir, stem + ".xml"))
        _, box       = biggest_box(xml_boxes)

        if aug:
            # More copies for small boxes — they are harder to learn
            area = _boxarea(box)
            t = nums + 3 if area < 0.125 else nums + 2 if area < 0.25 else nums
            for _ in range(t):
                try:
                    augmented = augmentor(image=img, bboxes=[box],
                                         class_labels=[LABEL_BLEED])
                    # BBoxSafeRandomCrop can drop the box entirely — skip if so.
                    # All three appends must be atomic: never append image
                    # unless box and label are also appended in the same branch.
                    if not augmented['bboxes']:
                        continue
                    aug_img = augmented['image']
                    aug_box = list(augmented['bboxes'][0])
                    aug_lbl = float(LABEL_BLEED)
                    images.append(aug_img)
                    boxes.append(aug_box)
                    labels.append(aug_lbl)
                except Exception:
                    pass
        else:
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

            if aug:
                for _ in range(nums):
                    try:
                        # Dummy full-image box required by BboxSafeRandomCrop
                        augmented = augmentor(image=img,
                                             bboxes=[[0., 0., 1., 1.]],
                                             class_labels=[LABEL_NONBLEED])
                        images.append(augmented['image'])
                        boxes.append([0.0, 0.0, 0.0, 0.0])   # no real box
                        labels.append(float(LABEL_NONBLEED))
                    except Exception:
                        pass
            else:
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

    aug=True multiplies the dataset via albumentations — essential
    for small datasets to prevent overfitting.

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

    if aug:
        augmentor = alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(),
            alb.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        ])

    # ── BLEEDING ONLY ────────────────────────────────
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        img_bgr = cv2.imread(os.path.join(img_dir, fname))
        if img_bgr is None:
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        # Masks use ann- prefix: "img- (1).png" → "ann- (1).png"
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

        if aug:
            for _ in range(nums):
                try:
                    augmented = augmentor(image=img, mask=mask)
                    images.append(augmented['image'])
                    masks.append(augmented['mask'])
                except Exception:
                    pass
        else:
            images.append(img)
            masks.append(mask)

    # Non-bleeding loop intentionally absent:
    # all-zero masks collapse U-Net Dice/IoU to 0.

    return (
        np.array(images, dtype=np.float32),
        np.array(masks,  dtype=np.float32),
    )