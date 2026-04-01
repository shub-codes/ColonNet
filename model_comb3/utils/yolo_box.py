"""
yolo_box.py  —  Combination 3
YOLOv8n bounding-box training + inference wrapper.

Why a separate file?
  YOLOv8n (ultralytics) has its own training loop and saves .pt weights.
  This module handles:
    1. prepare_yolo_dataset()   — converts our XML annotations to YOLO format
    2. train_yolo()             — calls ultralytics YOLO.train()
    3. YoloBoxPredictor         — loads the .pt model and produces
                                  normalised [x1, y1, x2, y2] boxes that
                                  match the format expected by Stage 4 (ColonNet)

Install:
    pip install ultralytics

Label convention: only bleeding images have valid boxes.
Non-bleeding images are skipped during YOLO dataset preparation.
"""

import os
import shutil
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# ── optional ultralytics import (guarded so the rest of the project
#    can still import this file even if ultralytics is not installed) ──
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

IMG_SIZE   = 224
IMG_EXTS   = (".png", ".jpg", ".jpeg", ".bmp")


# =====================================================
# 1. Dataset preparation: Pascal-VOC XML → YOLO TXT
# =====================================================

def _read_xml_box(xml_path, img_w, img_h):
    """
    Returns the biggest bounding box from a Pascal-VOC XML file as
    YOLO format: (cx, cy, w, h) normalised to [0, 1].
    Returns None if no valid box is found.
    """
    if not os.path.isfile(xml_path):
        return None
    try:
        root   = ET.parse(xml_path).getroot()
        size   = root.find("size")
        src_w  = float(size.find("width").text)
        src_h  = float(size.find("height").text)

        best_box, best_area = None, 0.0
        for obj in root.findall("object"):
            bb   = obj.find("bndbox")
            xmin = float(bb.find("xmin").text) / src_w
            ymin = float(bb.find("ymin").text) / src_h
            xmax = float(bb.find("xmax").text) / src_w
            ymax = float(bb.find("ymax").text) / src_h
            area = max(0, xmax - xmin) * max(0, ymax - ymin)
            if area > best_area:
                best_area = area
                best_box  = (xmin, ymin, xmax, ymax)

        if best_box is None or best_area <= 0:
            return None

        xmin, ymin, xmax, ymax = best_box
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        bw = xmax - xmin
        bh = ymax - ymin
        return (cx, cy, bw, bh)

    except Exception:
        return None


def prepare_yolo_dataset(data_root, yolo_root, val_split=0.2, seed=42):
    """
    Converts bleeding images + XML annotations into YOLO dataset format:

    yolo_root/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      dataset.yaml

    Only bleeding images with valid XML boxes are included.

    Parameters
    ----------
    data_root : str  path to TrainingDataset/
    yolo_root : str  output directory for the YOLO dataset
    val_split : float fraction of images used for validation
    seed      : int  random seed for reproducibility

    Returns
    -------
    yaml_path : str  path to dataset.yaml (pass to YOLO.train)
    """
    bleeding_path = os.path.join(data_root, "bleeding")
    img_dir       = os.path.join(bleeding_path, "Images")
    xml_dir       = os.path.join(bleeding_path, "Bounding boxes", "XML")

    # collect valid (image, box) pairs
    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        stem    = os.path.splitext(fname)[0]
        xml_p   = os.path.join(xml_dir, stem + ".xml")
        img_p   = os.path.join(img_dir, fname)
        box     = _read_xml_box(xml_p, IMG_SIZE, IMG_SIZE)
        if box is not None:
            pairs.append((img_p, box, stem))

    np.random.seed(seed)
    idx      = np.random.permutation(len(pairs))
    n_val    = max(1, int(len(pairs) * val_split))
    val_idx  = set(idx[:n_val].tolist())
    train_idx = set(idx[n_val:].tolist())

    for split, indices in [("train", train_idx), ("val", val_idx)]:
        img_out = os.path.join(yolo_root, "images", split)
        lbl_out = os.path.join(yolo_root, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for i in indices:
            img_p, (cx, cy, bw, bh), stem = pairs[i]

            # resize and save image as jpg
            img = cv2.imread(img_p)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            dst_img = os.path.join(img_out, stem + ".jpg")
            cv2.imwrite(dst_img, img)

            # YOLO label: class_id cx cy w h  (class 0 = bleeding)
            dst_lbl = os.path.join(lbl_out, stem + ".txt")
            with open(dst_lbl, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # write dataset.yaml
    yaml_path = os.path.join(yolo_root, "dataset.yaml")
    abs_root  = os.path.abspath(yolo_root)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_root}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['bleeding']\n")

    print(f"[YOLO dataset] {len(train_idx)} train / {len(val_idx)} val images")
    print(f"[YOLO dataset] YAML → {yaml_path}")
    return yaml_path


# =====================================================
# 2. YOLOv8n training
# =====================================================

def train_yolo(yaml_path, save_dir, epochs=50, imgsz=224,
               batch=16, patience=10, device=None):
    """
    Trains YOLOv8n on the prepared dataset.

    Parameters
    ----------
    yaml_path : str   path returned by prepare_yolo_dataset()
    save_dir  : str   directory where best.pt will be copied
    epochs    : int   max training epochs (early stopping via patience)
    imgsz     : int   input image size (must match IMG_SIZE)
    batch     : int   batch size
    patience  : int   early-stop patience (epochs without improvement)
    device    : str   "0" for GPU 0, "cpu", or None (auto)

    Returns
    -------
    best_pt : str  path to best.pt weights
    """
    if not _YOLO_AVAILABLE:
        raise ImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )

    os.makedirs(save_dir, exist_ok=True)

    model = YOLO("yolov8n.pt")   # downloads pretrained YOLOv8n on first run

    train_kwargs = dict(
        data     = yaml_path,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        patience = patience,
        project  = save_dir,
        name     = "yolo_bleed",
        exist_ok = True,
        verbose  = True,
    )
    if device is not None:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)

    # ultralytics saves best.pt under project/name/weights/best.pt
    trained_best = os.path.join(save_dir, "yolo_bleed", "weights", "best.pt")
    final_best   = os.path.join(save_dir, "yolo_best.pt")
    if os.path.exists(trained_best):
        shutil.copy2(trained_best, final_best)
        print(f"[YOLO] Best weights saved → {final_best}")
    else:
        final_best = trained_best   # fallback

    return final_best


# =====================================================
# 3. Inference wrapper → normalised [x1, y1, x2, y2]
# =====================================================

class YoloBoxPredictor:
    """
    Wraps a trained YOLOv8n model and returns bounding boxes in the
    same normalised [x1, y1, x2, y2] format used by the Keras pipeline.

    Usage
    -----
    predictor = YoloBoxPredictor("SavedModels/yolo_best.pt")
    boxes = predictor.predict_batch(images)   # images: (N, H, W, 3) float32 [0,1]
    # boxes: (N, 4) float32 normalised [x1,y1,x2,y2]
    """

    def __init__(self, weights_path, conf=0.25, imgsz=224):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        self.model = YOLO(weights_path)
        self.conf  = conf
        self.imgsz = imgsz

    def predict_single(self, image):
        """
        image : (H, W, 3) float32 in [0, 1]
        Returns normalised [x1, y1, x2, y2] float32 array.
        If no detection, returns [0, 0, 1, 1] (whole image).
        """
        img_uint8 = (image * 255).astype(np.uint8)
        results   = self.model(img_uint8, conf=self.conf,
                               imgsz=self.imgsz, verbose=False)
        boxes_xyxy = results[0].boxes.xyxyn.cpu().numpy()   # normalised xyxy

        if len(boxes_xyxy) == 0:
            return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        # pick box with highest confidence
        confs  = results[0].boxes.conf.cpu().numpy()
        best   = boxes_xyxy[np.argmax(confs)]
        return best.astype(np.float32)

    def predict_batch(self, images):
        """
        images : (N, H, W, 3) float32 in [0, 1]
        Returns (N, 4) float32 normalised boxes.
        """
        return np.stack([self.predict_single(img) for img in images], axis=0)