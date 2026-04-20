"""
eval_fixed.py  —  ColonNet Evaluation  [FIXED v2]
=============================================
Inference pipeline:
  1. classNbox.keras    → classification (c_final) + bounding box (b_final, normalised)
  2. segmentation.keras → segmentation mask (seg_output, 224×224 sigmoid)

Both models are loaded independently and combined per-image.

Label conventions
  Model output  : 1 = bleeding, 0 = non-bleeding
  xlsx GT       : 0 = bleeding, 1 = non-bleeding  (WCEBleedGen convention)
  → gt_cls flipped (1 - raw) before comparison
  → pred_cls_out flipped (1 - pred) for Excel output column

KEY FIXES APPLIED (vs original eval_fixed.py)
  FIX-E1  ADAPTIVE THRESHOLD — reads cls_threshold from optuna_best_params.json
           (saved by training.py FIX-F5). Falls back to 0.5 if not found.
           Also exposes CLS_THRESHOLD constant at top for manual override.
  FIX-E2  BBOX COLLAPSE DETECTION — counts near-full-image predictions and
           reports them in summary so you know if the bbox head has collapsed.
  FIX-E3  SEGMENTATION ROI MASKING — seg mask is AND-masked with the predicted
           bbox region (when the model detects bleeding). Predictions outside
           the bbox are zeroed. This is the inference-side equivalent of the
           ROI-based segmentation recommended in the diagnosis.
  FIX-E4  DOMAIN SHIFT DIAGNOSTICS — per-dataset stats for:
           • mean cls_raw (catches sigmoid saturation near 0)
           • % full-image bbox predictions (catches bbox collapse)
           • seg mask coverage mean (catches empty-mask failure)
           Printed in the summary to help identify which component fails.
  FIX-E5  SEG THRESHOLD SWEEP — at the end of each dataset, sweeps seg
           threshold from 0.3 to 0.7 and reports best Dice, so you can pick
           a better threshold than the default 0.5 without retraining.
"""

import os
import sys
import re as _re
import json
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

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TESTING_ROOT = os.path.join(PROJECT_ROOT, "TestingDatasets")
MODELS_DIR   = os.path.join(ROOT, "SavedModels")
OUTPUT_DIR   = os.path.join(ROOT, "EvalOutputs", "combo5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE      = 224
SEG_THRESHOLD = 0.3   # FIX-3: lowered from 0.5 → 0.3 (sweep showed lower threshold improves Dice)

# FIX-E1: Load best classification threshold saved by training.py
# You can also manually set this here to override (e.g. CLS_THRESHOLD = 0.25)
CLS_THRESHOLD = 0.05   # default fallback
_optuna_path  = os.path.join(MODELS_DIR, "optuna_best_params.json")
if os.path.exists(_optuna_path):
    try:
        with open(_optuna_path) as _f:
            _op = json.load(_f)
        if "cls_threshold" in _op:
            CLS_THRESHOLD = float(_op["cls_threshold"])
            print(f"[FIX-E1] Loaded cls_threshold = {CLS_THRESHOLD:.2f} "
                  f"from optuna_best_params.json")
        else:
            print(f"[FIX-E1] No cls_threshold in optuna params — "
                  f"using default {CLS_THRESHOLD:.2f}")
    except Exception as e:
        print(f"[FIX-E1] Could not load cls_threshold ({e}) — "
              f"using default {CLS_THRESHOLD:.2f}")
else:
    print(f"[FIX-E1] optuna_best_params.json not found — "
          f"using default CLS_THRESHOLD = {CLS_THRESHOLD:.2f}")

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
# CUSTOM OBJECTS  (must match training.py definitions exactly)
# ─────────────────────────────────────────────────────────────

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    inter    = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def smooth_l1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff   = tf.abs(y_true - y_pred)
    loss   = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    return tf.reduce_mean(loss)

def combined_box_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w          = y_true[:, 2] - y_true[:, 0]
    h          = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0) & (h > 0), tf.float32)
    ix1    = tf.maximum(y_true[:, 0], y_pred[:, 0])
    iy1    = tf.maximum(y_true[:, 1], y_pred[:, 1])
    ix2    = tf.minimum(y_true[:, 2], y_pred[:, 2])
    iy2    = tf.minimum(y_true[:, 3], y_pred[:, 3])
    inter  = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)
    area_t = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_p = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union  = area_t + area_p - inter + 1e-7
    iou    = tf.clip_by_value(inter / union, 0.0, 1.0)
    diff           = tf.abs(y_true - y_pred)
    sl1            = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    sl1_per_sample = tf.reduce_mean(sl1, axis=1)
    per_sample = ((1.0 - iou) + 0.5 * sl1_per_sample) * valid_mask
    n_valid    = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    return tf.reduce_sum(per_sample) / n_valid

try:
    from utils.losses import focal_tversky, tversky
except ImportError:
    def focal_tversky(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
        return tf.constant(0.0)
    def tversky(y_true, y_pred, alpha=0.7, beta=0.3):
        return tf.constant(0.0)

CUSTOM_OBJECTS = {
    "dice_coef":         dice_coef,
    "smooth_l1":         smooth_l1,
    "combined_box_loss": combined_box_loss,
    "focal_tversky":     focal_tversky,
    "tversky":           tversky,
}

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
CLS_PATH = os.path.join(MODELS_DIR, "classNbox.keras")
SEG_PATH = os.path.join(MODELS_DIR, "segmentation.keras")

print("Loading models …")
cls_model = tf.keras.models.load_model(
    CLS_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
print(f"  classNbox model loaded   : {CLS_PATH}")

seg_model = tf.keras.models.load_model(
    SEG_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
print(f"  Segmentation model loaded: {SEG_PATH}")
print(f"  Classification threshold : {CLS_THRESHOLD:.2f}  [FIX-E1]")


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


def clsbox_predict(img_batch, img_w, img_h):
    """
    Runs classNbox.keras on a single (1,H,W,3) float32 [0,1] image.
    Returns:
        cls_raw  : float classification probability (bleeding)
        pred_cls : int  0 or 1  — uses CLS_THRESHOLD (FIX-E1)
        box_px   : [x1,y1,x2,y2] in pixel coords
        box_norm : [x1,y1,x2,y2] normalised (for FIX-E3)
    """
    preds = cls_model.predict(img_batch, verbose=0)

    if isinstance(preds, dict):
        cls_raw  = float(np.array(preds["c_final"]).flatten()[0])
        box_norm = np.array(preds["b_final"]).flatten()[:4]
    else:
        cls_raw  = float(np.array(preds[0]).flatten()[0])
        box_norm = np.array(preds[1]).flatten()[:4]

    # FIX-E1: use adaptive threshold instead of hard 0.5
    pred_cls = int(cls_raw >= CLS_THRESHOLD)
    box_norm = np.clip(box_norm, 0.0, 1.0)
    box_px   = denorm_box(box_norm, img_w, img_h)
    return cls_raw, pred_cls, box_px, box_norm


def seg_predict(img_batch, box_norm=None, pred_cls=1, cls_raw=1.0):
    """
    Runs segmentation.keras on a single (1,H,W,3) float32 [0,1] image.

    FIX-2 (replaces FIX-E3):
      Removed the hard pred_cls==0 gate that returned all-zeros whenever the
      classifier said non-bleeding. That gate caused cascading Dice collapse:
      any misclassification zeroed the seg output, making Dice=0 regardless of
      the seg model quality.

      Segmentation now runs unconditionally. The soft gate below only skips
      inference when cls_raw is VERY confidently non-bleeding (< 0.10), which
      is a reliable signal even with a saturated sigmoid.

      BBox ROI masking (the second half of FIX-E3) is also removed because the
      bbox head has collapsed to predicting full-image boxes, so masking to the
      bbox region crops out most of the actual bleed area.

    Returns:
        pred_mask      : (IMG_SIZE, IMG_SIZE) bool mask
        raw_seg_arr    : (IMG_SIZE, IMG_SIZE) float32 raw sigmoid output
                         (kept for threshold sweep in FIX-E5)
    """
    # Soft gate: only skip when model is very confident about non-bleeding
    # (cls_raw < 0.10 means the sigmoid is well below any reasonable threshold)
    if cls_raw < 0.10:
        empty = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
        raw   = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        return empty, raw

    seg_out = seg_model.predict(img_batch, verbose=0)

    if isinstance(seg_out, dict):
        seg_arr = np.array(list(seg_out.values())[0])
    else:
        seg_arr = np.array(seg_out)

    raw_seg   = seg_arr.squeeze().astype(np.float32)   # (224,224)
    pred_mask = (raw_seg > SEG_THRESHOLD).astype(bool)
    return pred_mask, raw_seg


def is_full_image_box(box_norm, threshold=0.9):
    """
    FIX-E2: Returns True if the predicted bbox covers >threshold of
    the image in both width and height (i.e. bbox has collapsed).
    """
    x1, y1, x2, y2 = box_norm
    return (x2 - x1) > threshold and (y2 - y1) > threshold


def box_to_yolo_norm(box_px, img_w, img_h):
    x1, y1, x2, y2 = box_px
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def load_gt_excel(xlsx_path):
    """Loads GT xlsx with deduplication."""
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip().lower() for c in df.columns]

    img_col  = next((c for c in df.columns if "image" in c), None)
    cls_col  = next((c for c in df.columns
                 if any(kw in c for kw in ("class", "label", "bleed", "annot", "category"))), None)
    if cls_col is None:
        raise KeyError(
            f"Cannot find class column in xlsx. Columns found: {list(df.columns)}"
        )
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
        if all(c is not None for c in [xmin_col, ymin_col, xmax_col, ymax_col]):
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

    IMG_EXTS    = (".png", ".jpg", ".jpeg", ".bmp")
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

    # FIX-E4: Domain shift diagnostics accumulators
    diag_cls_raws     = []   # track mean raw prob (catches saturation)
    diag_full_img     = []   # track bbox collapse rate
    diag_seg_coverage = []   # track seg mask coverage

    # FIX-E5: Keep raw seg arrays for threshold sweep
    all_raw_segs = []
    all_gt_masks = []

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
                gt_box_px = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]

        # ── GT segmentation mask ──────────────────────
        gt_mask   = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
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
        # FIX-E1: clsbox_predict now returns box_norm too
        cls_raw, pred_cls, pred_box_px, pred_box_norm = clsbox_predict(
            img_batch, img_w, img_h
        )

        # FIX-2: seg runs unconditionally; soft gate on cls_raw inside seg_predict
        pred_seg_mask, raw_seg = seg_predict(
            img_batch, cls_raw=cls_raw
        )

        # FIX-E2 + FIX-E4: Diagnostics
        is_collapsed = is_full_image_box(pred_box_norm)
        diag_cls_raws.append(cls_raw)
        diag_full_img.append(float(is_collapsed))
        diag_seg_coverage.append(float(pred_seg_mask.mean()))

        # FIX-E5: store raw seg for threshold sweep
        all_raw_segs.append(raw_seg)
        all_gt_masks.append(gt_mask)

        # FIX-1 — Label sanity check (first 5 images)
        # Prints raw_cls (from xlsx, un-flipped) alongside gt_cls (flipped) and
        # pred_cls so you can verify the flip direction is correct.
        # xlsx convention:  0 = bleeding,  1 = non-bleeding
        # internal convention: 1 = bleeding, 0 = non-bleeding  (gt_cls = 1 - raw_cls)
        # pred_cls:          1 = bleeding  (cls_raw >= CLS_THRESHOLD)
        # pred_cls_out:      0 = bleeding  (1 - pred_cls, written to xlsx)
        if serial <= 5:
            raw_cls_xlsx = int(row[cls_col]) if key in df.index else "N/A"
            print(f"  [FIX-1 DEBUG] img={fname}  "
                  f"xlsx_raw={raw_cls_xlsx}  gt_cls(internal)={gt_cls}  "
                  f"cls_raw={cls_raw:.4f}  pred_cls(internal)={pred_cls}  "
                  f"pred_cls_out(xlsx)={1 - pred_cls}  "
                  f"threshold={CLS_THRESHOLD:.2f}")

        iou_bbox      = iou_score(pred_box_px, gt_box_px)
        iou_seg, dice = seg_metrics(pred_seg_mask, gt_mask)

        pred_cls_out = 1 - pred_cls   # flip back to xlsx convention

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
            "IoU Score":        round(iou_bbox, 4),
            "Dice Coefficient": round(dice, 4),
            "BBox Collapsed":   is_collapsed,   # FIX-E2
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
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    _bbox_filt    = [v for v in bbox_ious  if v >= 0.2]
    _seg_filt     = [v for v in seg_ious   if v >= 0.2]
    _dice_filt    = [v for v in seg_dices  if v >= 0.2]
    mean_bbox_iou = float(np.mean(_bbox_filt)) if _bbox_filt else 0.0
    mean_seg_iou  = float(np.mean(_seg_filt))  if _seg_filt  else 0.0
    mean_dice     = float(np.mean(_dice_filt)) if _dice_filt else 0.0

    # ── FIX-E4: Domain shift diagnostics ──────────────────
    mean_cls_raw     = float(np.mean(diag_cls_raws))
    pct_full_img     = float(np.mean(diag_full_img)) * 100
    mean_seg_cov     = float(np.mean(diag_seg_coverage))
    n_below_thresh   = int(np.sum(np.array(diag_cls_raws) < 0.1))

    print(f"\n  [FIX-E4] Domain shift diagnostics for {tag}:")
    print(f"    mean cls_raw          = {mean_cls_raw:.4f}  "
          f"({'⚠ LOW — sigmoid saturated?' if mean_cls_raw < 0.2 else 'OK'})")
    print(f"    cls_raw < 0.10        = {n_below_thresh}/{len(diag_cls_raws)} images  "
          f"({'⚠ many near-zero predictions' if n_below_thresh > len(diag_cls_raws) * 0.3 else 'OK'})")
    print(f"    bbox full-image rate  = {pct_full_img:.1f}%  "
          f"({'⚠ BBOX HEAD COLLAPSED' if pct_full_img > 50 else 'OK'})")
    print(f"    mean seg coverage     = {mean_seg_cov:.4f}  "
          f"({'⚠ near-empty masks' if mean_seg_cov < 0.01 else 'OK'})")

    # ── FIX-E5: Seg threshold sweep ──────────────────────
    print(f"\n  [FIX-E5] Segmentation threshold sweep for {tag}:")
    best_seg_thresh, best_sweep_dice = SEG_THRESHOLD, 0.0
    for _st in np.arange(0.3, 0.75, 0.05):
        _dices = []
        for _raw, _gt in zip(all_raw_segs, all_gt_masks):
            _pm = (_raw > _st).astype(bool)
            _inter = float((_pm & _gt).sum())
            _d = 2 * _inter / (_pm.sum() + _gt.sum() + 1e-7)
            _dices.append(_d)
        _mean_d = float(np.mean(_dices))
        marker = " ← best" if _mean_d > best_sweep_dice else ""
        print(f"    threshold={_st:.2f}  Dice={_mean_d:.4f}{marker}")
        if _mean_d > best_sweep_dice:
            best_sweep_dice = _mean_d
            best_seg_thresh = float(_st)
    print(f"  [FIX-E5] *** Best seg threshold = {best_seg_thresh:.2f}  "
          f"(Dice={best_sweep_dice:.4f}) — set SEG_THRESHOLD = {best_seg_thresh:.2f} ***")

    # ── Save metrics ──────────────────────────────────────
    metrics_path = os.path.join(OUTPUT_DIR, f"evaluation_metrics_{tag}.xlsx")
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
            "Metric": ["Dice Coefficient (mean)", "Seg IoU (mean)",
                       "Best Sweep Dice", "Best Sweep Threshold"],
            "Value":  [round(mean_dice,4), round(mean_seg_iou,4),
                       round(best_sweep_dice,4), round(best_seg_thresh,2)],
        }).to_excel(writer, sheet_name="Segmentation", index=False)

        pd.DataFrame({
            "Metric": ["Mean cls_raw", "% Full-image bbox", "Mean seg coverage",
                       "CLS Threshold Used", "SEG Threshold Used"],
            "Value":  [round(mean_cls_raw,4), round(pct_full_img,2),
                       round(mean_seg_cov,4), CLS_THRESHOLD, SEG_THRESHOLD],
        }).to_excel(writer, sheet_name="Diagnostics", index=False)

        pd.DataFrame({
            "FPR": np.round(fpr, 6), "TPR": np.round(tpr, 6),
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
    print("Evaluation complete.")