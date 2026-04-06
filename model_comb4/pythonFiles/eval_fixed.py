"""
combo4_eval.py  —  Combination 4 Evaluation
=============================================
Inference pipeline:
  1. EfficientNet-B0  → classification (bleeding probability)
  2. YOLOv8n-seg      → bounding box + instance segmentation mask

Both models are loaded independently and combined per-image.

Label conventions
  Model output  : 1 = bleeding, 0 = non-bleeding
  xlsx GT       : 0 = bleeding, 1 = non-bleeding  (WCEBleedGen convention)
  → gt_cls flipped (1 - raw) before comparison (same as Combo 3 eval)
  → pred_cls_out flipped (1 - pred) for Excel output column
"""

import os
import sys
import re as _re
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    average_precision_score, roc_auc_score, roc_curve,
)

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TESTING_ROOT = os.path.join(PROJECT_ROOT, "TestingDatasets")
MODELS_DIR   = os.path.join(ROOT, "SavedModels")
OUTPUT_DIR   = os.path.join(ROOT, "EvalOutputs", "combo4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
YOLO_CONF_THRESHOLD = 0.25   # YOLO confidence threshold for detection

DATASETS = {
    "TestDataset1": {
        "root":       os.path.join(TESTING_ROOT, "Test Dataset 1"),
        "images_dir": "Unmarked Images",
        "xlsx":       "Test Dataset 1 TXT (True labels).xlsx",
    },
    "TestDataset2": {
        "root":       os.path.join(TESTING_ROOT, "Test Dataset 2"),
        "images_dir": "Images",
        "xlsx":       "Test Dataset 2 TXT (True labels).xlsx",
    },
}

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    inter    = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

CLS_PATH  = os.path.join(MODELS_DIR, "combo4_classifier.keras")
YOLO_PATH = os.path.join(MODELS_DIR, "combo4_yolo_best.pt")

print("Loading Combo 4 models …")
cls_model = tf.keras.models.load_model(
    CLS_PATH, custom_objects={"dice_coef": dice_coef}, compile=False)
print(f"  Classifier loaded: {CLS_PATH}")

yolo_model = None
if _YOLO_AVAILABLE and os.path.exists(YOLO_PATH):
    yolo_model = _YOLO(YOLO_PATH)
    print(f"  YOLOv8n-seg loaded: {YOLO_PATH}")
else:
    print("  WARNING: YOLOv8n-seg not available — box/seg metrics will be 0.")


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def load_image(path):
    """Returns (PIL image, (1,224,224,3) float32 batch in [0,1])."""
    pil = Image.open(path).convert("RGB")
    arr = np.array(pil.resize((IMG_SIZE, IMG_SIZE),
                               Image.BILINEAR)).astype(np.float32) / 255.0
    return pil, arr[np.newaxis]


def iou_score(box_pred, box_true):
    """Both boxes in pixel coords [x1,y1,x2,y2]. Returns IoU in [0,1]."""
    if (box_pred[2] <= box_pred[0] or box_pred[3] <= box_pred[1] or
            box_true[2] <= box_true[0] or box_true[3] <= box_true[1]):
        return 0.0
    ix1 = max(box_pred[0], box_true[0])
    iy1 = max(box_pred[1], box_true[1])
    ix2 = min(box_pred[2], box_true[2])
    iy2 = min(box_pred[3], box_true[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    ap    = (box_pred[2]-box_pred[0]) * (box_pred[3]-box_pred[1])
    at    = (box_true[2]-box_true[0]) * (box_true[3]-box_true[1])
    union = ap + at - inter + 1e-7
    return float(np.clip(inter / union, 0, 1))


def seg_metrics(pred_mask, gt_mask):
    """Returns (IoU, Dice) for boolean masks."""
    inter = float((pred_mask & gt_mask).sum())
    union = float((pred_mask | gt_mask).sum())
    iou   = inter / (union + 1e-7)
    dice  = 2 * inter / (pred_mask.sum() + gt_mask.sum() + 1e-7)
    return iou, dice


def denorm_box(box_norm, img_w, img_h):
    """Converts normalised [x1,y1,x2,y2] → pixel [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = np.array(box_norm).flatten()[:4]
    return [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]


def yolo_predict(img_arr_01, img_w, img_h):
    """
    Runs YOLOv8n-seg on a single (1,H,W,3) float32 [0,1] image.
    Returns:
        box_px   : [x1,y1,x2,y2] in pixel coords
        seg_mask : (IMG_SIZE, IMG_SIZE) bool mask
        conf     : float detection confidence (0 if no detection)
    """
    no_box  = [0.0, 0.0, 0.0, 0.0]
    no_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)

    if yolo_model is None:
        return no_box, no_mask, 0.0

    img_uint8 = (img_arr_01.squeeze() * 255).astype(np.uint8)
    results   = yolo_model(img_uint8, conf=YOLO_CONF_THRESHOLD,
                           imgsz=IMG_SIZE, verbose=False)

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return no_box, no_mask, 0.0

    # Pick detection with highest confidence
    confs      = r.boxes.conf.cpu().numpy()
    best_idx   = int(np.argmax(confs))
    best_conf  = float(confs[best_idx])

    # Normalised xyxy → pixel coords
    xyxy_norm = r.boxes.xyxyn.cpu().numpy()[best_idx]
    box_px    = [xyxy_norm[0] * img_w, xyxy_norm[1] * img_h,
                 xyxy_norm[2] * img_w, xyxy_norm[3] * img_h]

    # Segmentation mask
    seg_mask = no_mask
    if r.masks is not None:
        try:
            # masks.data is (N, H, W) float tensor
            mask_data = r.masks.data.cpu().numpy()[best_idx]
            # Resize to IMG_SIZE
            import cv2
            mask_resized = cv2.resize(mask_data, (IMG_SIZE, IMG_SIZE),
                                      interpolation=cv2.INTER_NEAREST)
            seg_mask = (mask_resized > 0.5).astype(bool)
        except Exception:
            pass

    return box_px, seg_mask, best_conf


def box_to_yolo_norm(box_px, img_w, img_h):
    x1, y1, x2, y2 = box_px
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def load_gt_excel(xlsx_path):
    """Loads GT xlsx with deduplication (same as Combo 3 eval)."""
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Find column names flexibly
    img_col  = next((c for c in df.columns if "image" in c), None)
    cls_col  = next((c for c in df.columns
                     if c in ("class", "label", "category")), "class")
    xmin_col = next((c for c in df.columns if "xmin" in c or "x_min" in c), None)
    ymin_col = next((c for c in df.columns if "ymin" in c or "y_min" in c), None)
    xmax_col = next((c for c in df.columns if "xmax" in c or "x_max" in c), None)
    ymax_col = next((c for c in df.columns if "ymax" in c or "y_max" in c), None)

    if img_col:
        df[img_col] = df[img_col].astype(str).str.strip()
        df["_key"]  = df[img_col].apply(lambda s: os.path.splitext(s)[0])
        df = df.drop_duplicates(subset="_key").set_index("_key")

    return df, cls_col, xmin_col, ymin_col, xmax_col, ymax_col


def get_gt_box(row, xmin_col, ymin_col, xmax_col, ymax_col):
    """Returns normalised [x1,y1,x2,y2] or None."""
    try:
        if all(c is not None for c in [xmin_col,ymin_col,xmax_col,ymax_col]):
            return [float(row[xmin_col]), float(row[ymin_col]),
                    float(row[xmax_col]), float(row[ymax_col])]
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_dataset(tag, cfg):
    print(f"\n{'='*55}")
    print(f"Evaluating {tag}")
    print(f"{'='*55}")

    root       = cfg["root"]
    img_dir    = os.path.join(root, cfg["images_dir"])
    xlsx_path  = os.path.join(root, cfg["xlsx"])
    annot_dir  = os.path.join(root, "Annotations")

    IMG_EXTS   = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(IMG_EXTS)
    ])
    print(f"  Images found: {len(image_files)}")

    df, cls_col, xmin_col, ymin_col, xmax_col, ymax_col = \
        load_gt_excel(xlsx_path)

    rows_txt, rows_yolo = [], []
    cls_preds, cls_gts, cls_probs = [], [], []
    bbox_ious, seg_ious, seg_dices = [], [], []

    for serial, img_path in enumerate(tqdm(image_files, desc=tag), start=1):
        fname = os.path.basename(img_path)
        key   = os.path.splitext(fname)[0]

        pil_img, img_batch = load_image(img_path)
        img_w, img_h = pil_img.size

        # ── Ground truth ──────────────────────────────
        gt_cls    = -1
        gt_box_px = [0.0, 0.0, 0.0, 0.0]

        if key in df.index:
            row = df.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            raw_cls = int(row[cls_col])
            gt_cls  = 1 - raw_cls   # flip: xlsx 0=bleed → model 1=bleed
            box     = get_gt_box(row, xmin_col, ymin_col, xmax_col, ymax_col)
            if box is not None:
                gt_box_px = denorm_box(box, img_w, img_h)

        # ── GT segmentation mask ──────────────────────
        gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
        mask_path = None
        for ext in (".png", ".bmp", ".jpg", ".tif"):
            c = os.path.join(annot_dir, key + ext)
            if os.path.exists(c):
                mask_path = c; break
        if mask_path is None:
            kn = _re.search(r"\d+", key)
            if kn:
                for af in os.listdir(annot_dir):
                    an = _re.search(r"\d+", af)
                    if an and an.group() == kn.group():
                        mask_path = os.path.join(annot_dir, af); break
        if mask_path:
            gt_mask = (np.array(
                Image.open(mask_path).convert("L")
                     .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            ) > 127).astype(bool)

        # ── Predictions ───────────────────────────────
        # Classification from EfficientNet-B0
        cls_raw  = float(cls_model.predict(img_batch, verbose=0).flatten()[0])
        pred_cls = int(round(cls_raw))

        # Box + segmentation from YOLOv8n-seg
        pred_box_px, pred_seg_mask, yolo_conf = yolo_predict(
            img_batch, img_w, img_h)

        # When YOLO has no detection, use classifier confidence as proxy
        # (no box / mask predicted → IoU will be 0, which is correct)
        if serial == 1:
            print(f"  [DEBUG] gt_cls={gt_cls}  cls_raw={cls_raw:.4f}  "
                  f"pred_cls={pred_cls}  yolo_conf={yolo_conf:.4f}")
            print(f"  [DEBUG] pred_box={[round(v,1) for v in pred_box_px]}")
            print(f"  [DEBUG] gt_box ={[round(v,1) for v in gt_box_px]}")

        iou_bbox      = iou_score(pred_box_px, gt_box_px)
        iou_seg, dice = seg_metrics(pred_seg_mask, gt_mask)

        # xlsx convention output (0=bleeding)
        pred_cls_out = 1 - pred_cls

        bbox_ious.append(iou_bbox);  seg_ious.append(iou_seg)
        seg_dices.append(dice);      cls_preds.append(pred_cls)
        cls_gts.append(gt_cls);      cls_probs.append(cls_raw)

        rows_txt.append({
            "Serial Number":    serial,
            "Image Number":     fname,
            "Predicted Class":  pred_cls_out,
            "x_min": round(pred_box_px[0], 2),
            "y_min": round(pred_box_px[1], 2),
            "x_max": round(pred_box_px[2], 2),
            "y_max": round(pred_box_px[3], 2),
            "Confidence Score": round(cls_raw, 4),
            "YOLO Conf":        round(yolo_conf, 4),
            "IoU Score":        round(iou_bbox, 4),
            "Dice Coefficient": round(dice, 4),
        })

        cx, cy, bw, bh = box_to_yolo_norm(pred_box_px, img_w, img_h)
        rows_yolo.append({
            "Serial Number":    serial,
            "Image Number":     fname,
            "Predicted Class":  pred_cls_out,
            "x_mid": round(cx, 6), "y_mid": round(cy, 6),
            "width": round(bw, 6), "height": round(bh, 6),
            "Confidence Score": round(cls_raw, 4),
            "IoU Score":        round(iou_bbox, 4),
            "Dice Coefficient": round(dice, 4),
        })

    pd.DataFrame(rows_txt).to_excel(
        os.path.join(OUTPUT_DIR, f"{tag}_predictions_txt.xlsx"), index=False)
    pd.DataFrame(rows_yolo).to_excel(
        os.path.join(OUTPUT_DIR, f"{tag}_predictions_yolo.xlsx"), index=False)

    valid   = [i for i, g in enumerate(cls_gts) if g != -1]
    gts_v   = [cls_gts[i]   for i in valid]
    preds_v = [cls_preds[i] for i in valid]
    probs_v = [cls_probs[i] for i in valid]

    if not gts_v:
        print(f"\n  WARNING: No GT labels matched for {tag}.")
        return

    acc = accuracy_score(gts_v, preds_v)
    rec = recall_score(gts_v, preds_v, average="macro", zero_division=0)
    f1  = f1_score(gts_v, preds_v, average="macro", zero_division=0)
    try:    ap = average_precision_score(gts_v, probs_v)
    except: ap = float("nan")
    try:
        auc = roc_auc_score(gts_v, probs_v)
        fpr, tpr, _ = roc_curve(gts_v, probs_v)
    except Exception:
        auc = float("nan")
        fpr, tpr = np.array([0.0,1.0]), np.array([0.0,1.0])

    mean_bbox_iou = float(np.mean(bbox_ious))
    mean_seg_iou  = float(np.mean(seg_ious))
    mean_dice     = float(np.mean(seg_dices))

    metrics_path = os.path.join(OUTPUT_DIR,
                                f"evaluation_metrics_{tag}.xlsx")
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        pd.DataFrame({
            "Metric": ["Accuracy", "Recall", "F1-Score", "ROC-AUC"],
            "Value":  [round(acc,4), round(rec,4), round(f1,4), round(auc,4)],
        }).to_excel(writer, sheet_name="Classification", index=False)

        pd.DataFrame({
            "Metric": ["Average Precision", "BBox IoU (mean)"],
            "Value":  [round(ap,4), round(mean_bbox_iou,4)],
        }).to_excel(writer, sheet_name="Detection", index=False)

        pd.DataFrame({
            "Metric": ["Dice Coefficient (mean)", "Seg IoU (mean)"],
            "Value":  [round(mean_dice,4), round(mean_seg_iou,4)],
        }).to_excel(writer, sheet_name="Segmentation", index=False)

        pd.DataFrame({
            "FPR": np.round(fpr,6), "TPR": np.round(tpr,6),
        }).to_excel(writer, sheet_name="ROC Curve Data", index=False)

    print(f"\n  Results for {tag}")
    print(f"  {'─'*42}")
    print(f"  Classification   Acc={acc:.4f}  Rec={rec:.4f}  "
          f"F1={f1:.4f}  AUC={auc:.4f}")
    print(f"  Detection        AP={ap:.4f}   BBox-IoU={mean_bbox_iou:.4f}")
    print(f"  Segmentation     Dice={mean_dice:.4f}  "
          f"Seg-IoU={mean_seg_iou:.4f}")
    print(f"  {'─'*42}")
    print(f"  Outputs → {OUTPUT_DIR}\n")


if __name__ == "__main__":
    for tag, cfg in DATASETS.items():
        evaluate_dataset(tag, cfg)
    print("Combo 4 evaluation complete.")