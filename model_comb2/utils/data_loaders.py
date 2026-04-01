import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

IMG_SIZE = 224

# Label convention
LABEL_BLEED    = 0
LABEL_NONBLEED = 1

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_DEFAULT_DATA = r"C:\Users\Shubham\Desktop\ColonNet\TrainingDataset"


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
    if data_root is None:
        data_root = _DEFAULT_DATA

    bleeding_path     = os.path.join(data_root, "bleeding")
    non_bleeding_path = os.path.join(data_root, "non-bleeding")

    xml_dir = os.path.join(bleeding_path, "Bounding boxes", "XML")
    img_dir = os.path.join(bleeding_path, "Images")

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    images, boxes, labels, valid_mask = [], [], [], []

    # ── BLEEDING ──
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue

        img_bgr = cv2.imread(os.path.join(img_dir, fname))
        if img_bgr is None:
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        stem = os.path.splitext(fname)[0]
        xml_boxes, _ = read_xml(os.path.join(xml_dir, stem + ".xml"))
        found, box   = biggest_box(xml_boxes)

        images.append(img)
        boxes.append(box)
        # bleeding samples
        labels.append(1.0 - float(LABEL_BLEED))    # 0 → 1


        valid_mask.append(1.0)  # valid bbox

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
            boxes.append([0.0, 0.0, 0.0, 0.0])  # dummy
            # non-bleeding samples  
            labels.append(1.0 - float(LABEL_NONBLEED)) # 1 → 0  
            valid_mask.append(0.0)  # IMPORTANT

    return (
        np.array(images, dtype=np.float32),
        np.array(boxes,  dtype=np.float32),
        np.array(labels, dtype=np.float32),
        # np.array(valid_mask, dtype=np.float32),  # NEW
    )


# =====================================================
# SEGMENTATION LOADER
# =====================================================

def load_data_unet(aug=False, nums=1, data_root=None, include_neg=False):
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

        stem      = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, stem + ".png")

        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, stem + ".bmp")

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