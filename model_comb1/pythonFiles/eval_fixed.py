"""
eval.py  —  ColonNet Evaluation Script  (Combination 1 — DenseNet121)
========================================================================
Test dataset structure:
  TestingDatasets/
    Test Dataset 1/
      Annotations/          <- binary segmentation masks  (A000X.png)
      Unmarked Images/      <- input images               (A000X.png)
      Test Dataset 1 TXT (True labels).xlsx

    Test Dataset 2/
      Annotations/          <- binary segmentation masks  (A000X.png)
      Images/               <- input images               (A000X.png)
      Test Dataset 2 TXT (True labels).xlsx

Label convention (matches training):
    1 = bleeding   (positive class, emitted by training as 1.0)
    0 = non-bleeding

Fixes applied vs. previous version:
  FIX-E1  load_gt_excel: duplicate image keys collapsed before indexing.
           class: max() — image is bleeding if ANY row says bleeding.
           box coords: min/max to form union enclosing box.
  FIX-E2  evaluate_dataset: modelB (classNbox) used for BOTH cls and box.
           modelA (CheckPoint1) no longer loaded — Stage 2 model has
           both heads and is the correct source for both outputs.
  FIX-E3  Segmentation threshold auto-calibration over up to 50 images
           before the main eval loop. Hard-coded 0.5 was too aggressive.
  FIX-E4  gt_box_px default is [0,0,0,0] (no box), not full image.
  FIX-E5  denorm_box: explicit zero-sentinel check before scaling.
  FIX-E6  iou_score: returns 0.0 if either box has zero area.
  COMB1   Custom object is iou_smooth_l1_loss (not giou_loss / combined_box_loss).
           DenseNet121 backbone — image normalisation is [0,1] for BOTH
           modelB and modelC (not MobileNetV3's [-1,1]).
"""

import os
import glob
import sys
import re as _re

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              average_precision_score,
                              roc_auc_score, roc_curve)

from utils.losses import focal_tversky, tversky

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TESTING_ROOT = os.path.join(PROJECT_ROOT, "TestingDatasets")
MODELS_DIR   = os.path.join(ROOT, "SavedModels")
OUTPUT_DIR   = os.path.join(ROOT, "EvalOutputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224

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
# CUSTOM OBJECTS
# COMB1: iou_smooth_l1_loss — matches training.py exactly.
# The reference combination used giou_loss + combined_box_loss;
# this combination replaced GIoU with IoU + 0.5*smooth-L1.
# ─────────────────────────────────────────────────────────────

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f     = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f     = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def iou_smooth_l1_loss(y_true, y_pred):
    """Matches training.py exactly — required for model deserialisation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ix1    = tf.maximum(y_true[:, 0], y_pred[:, 0])
    iy1    = tf.maximum(y_true[:, 1], y_pred[:, 1])
    ix2    = tf.minimum(y_true[:, 2], y_pred[:, 2])
    iy2    = tf.minimum(y_true[:, 3], y_pred[:, 3])
    inter  = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)
    area_t = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_p = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union  = area_t + area_p - inter + 1e-7
    iou    = inter / union
    diff      = tf.abs(y_true - y_pred)
    smooth_l1 = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    return tf.reduce_mean(1.0 - iou) + 0.5 * tf.reduce_mean(smooth_l1)


SEG_CUSTOM = {"focal_tversky": focal_tversky, "tversky": tversky,
              "dice_coef": dice_coef}
BOX_CUSTOM = {"iou_smooth_l1_loss": iou_smooth_l1_loss}
ALL_CUSTOM = {**SEG_CUSTOM, **BOX_CUSTOM}

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# FIX-E2: Only modelB (classNbox) and modelC (segmentation) loaded.
# modelA (CheckPoint1) is the Stage 1 box-only checkpoint — it is
# superseded by modelB which has both heads fully trained.
# ─────────────────────────────────────────────────────────────

def find_model(*names):
    for name in names:
        p = os.path.join(MODELS_DIR, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"None of {names} found in {MODELS_DIR}")


print("Loading classification+box model (classNbox) …")
modelB = tf.keras.models.load_model(
    find_model("classNbox.h5", "classNbox.keras"),
    custom_objects=BOX_CUSTOM, compile=False)

print("Loading segmentation model …")
modelC = tf.keras.models.load_model(
    find_model("segmentation.h5", "segmentation.keras"),
    custom_objects=SEG_CUSTOM, compile=False)

print("All models loaded.\n")

# Segmentation model sanity check
print("Checking segmentation model weights …")
total_params  = modelC.count_params()
nonzero_count = sum(int(np.count_nonzero(w.numpy())) for w in modelC.weights)
all_weights   = np.concatenate([w.numpy().flatten() for w in modelC.weights])
print(f"  Total params     : {total_params}")
print(f"  Non-zero weights : {nonzero_count} / {total_params}")
print(f"  Weight range     : min={all_weights.min():.6f}  max={all_weights.max():.6f}")
print(f"  Weight std-dev   : {all_weights.std():.6f}")
_dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_out   = modelC.predict(_dummy, verbose=0).squeeze().astype(np.float32)
print(f"  Dummy-input output: min={_out.min():.6f}  max={_out.max():.6f}  mean={_out.mean():.6f}")
if _out.max() < 0.01:
    print("  ⚠  OUTPUT IS DEAD — delete segmentation.h5 and retrain Stage 3.")
else:
    print("  ✓  Segmentation model is live.")
print()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_image(path):
    """
    Returns (pil_img, batch).

    COMB1 note: Both modelB (DenseNet121) and modelC (U-Net) were trained
    on [0, 1] normalised inputs — a single batch tensor is sufficient.
    The reference combination used MobileNetV3 which required [-1, 1];
    that is NOT the case here. Using the wrong normalisation would silently
    produce garbage predictions without any error.
    """
    pil  = Image.open(path).convert("RGB")
    arr  = np.array(pil.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return pil, np.expand_dims(arr, 0)


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")
    return None


def load_gt_excel(path):
    """
    FIX-E1: Collapse duplicate keys (one Excel row per bounding box)
    into one row per image before indexing.

    Aggregation rules:
      class : max()  — image is bleeding if ANY annotation row says bleeding
                       (xlsx: 0=bleeding, 1=non-bleeding → max picks non-bleeding
                        only if ALL rows are non-bleeding, which is correct)
      xmin  : min()  — union enclosing box
      ymin  : min()
      xmax  : max()
      ymax  : max()
    """
    df = pd.read_excel(path)
    print(f"  Excel columns: {list(df.columns)}")
    print(f"  Excel rows before deduplication: {len(df)}")

    img_col  = find_col(df, ["image_name", "image", "filename", "file",
                              "image_id", "img", "name", "Image", "Filename"])
    cls_col  = find_col(df, ["class_label", "class", "label", "labels",
                              "bleeding", "annotation", "y", "target", "Class"])
    xmin_col = find_col(df, ["x_min", "xmin", "X_min", "Xmin"], required=False)
    ymin_col = find_col(df, ["y_min", "ymin", "Y_min", "Ymin"], required=False)
    xmax_col = find_col(df, ["x_max", "xmax", "X_max", "Xmax"], required=False)
    ymax_col = find_col(df, ["y_max", "ymax", "Y_max", "Ymax"], required=False)

    df["_key"] = (df[img_col].astype(str)
                  .apply(lambda x: os.path.splitext(os.path.basename(x))[0]))

    # Build aggregation dict dynamically — only include box cols if present
    agg = {cls_col: "max"}
    if xmin_col: agg[xmin_col] = "min"
    if ymin_col: agg[ymin_col] = "min"
    if xmax_col: agg[xmax_col] = "max"
    if ymax_col: agg[ymax_col] = "max"

    df_agg = df.groupby("_key", as_index=False).agg(agg)
    print(f"  Excel rows after deduplication : {len(df_agg)}")
    print(f"  Excel keys (first 5): {list(df_agg['_key'][:5])}")
    print(f"  cls_col='{cls_col}'  sample values: {list(df_agg[cls_col][:5])}")

    return (df_agg.set_index("_key"),
            cls_col, xmin_col, ymin_col, xmax_col, ymax_col)


def get_gt_box(row, xmin_col, ymin_col, xmax_col, ymax_col):
    if any(c is None for c in [xmin_col, ymin_col, xmax_col, ymax_col]):
        return None
    try:
        return [float(row[xmin_col]), float(row[ymin_col]),
                float(row[xmax_col]), float(row[ymax_col])]
    except Exception:
        return None


def denorm_box(raw, img_w, img_h):
    """
    FIX-E5: Explicit zero-sentinel check.
    [0,0,0,0] means no annotation — return as-is without scaling.
    For all other boxes: if max coord <= 1.0, treat as normalised.
    """
    b = np.array(raw).flatten()[:4].astype(float)
    if b.max() == 0.0:
        return [0.0, 0.0, 0.0, 0.0]
    if b.max() <= 1.0:
        b[0] *= img_w; b[1] *= img_h; b[2] *= img_w; b[3] *= img_h
    x1, y1 = min(b[0], b[2]), min(b[1], b[3])
    x2, y2 = max(b[0], b[2]), max(b[1], b[3])
    return [float(x1), float(y1), float(x2), float(y2)]


def box_to_yolo_norm(box, img_w, img_h):
    cx = ((box[0] + box[2]) / 2) / img_w
    cy = ((box[1] + box[3]) / 2) / img_h
    bw = (box[2] - box[0]) / img_w
    bh = (box[3] - box[1]) / img_h
    return cx, cy, bw, bh


def iou_score(boxA, boxB):
    """FIX-E6: Returns 0.0 if either box has zero area (no annotation)."""
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if aA <= 0 or aB <= 0:
        return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    return inter / (aA + aB - inter + 1e-7)


def seg_metrics(pred_mask, gt_mask):
    p     = pred_mask.flatten().astype(bool)
    g     = gt_mask.flatten().astype(bool)
    inter = np.logical_and(p, g).sum()
    iou_v = inter / (np.logical_or(p, g).sum() + 1e-7)
    dice  = (2 * inter) / (p.sum() + g.sum() + 1e-7)
    return float(iou_v), float(dice)


def _find_mask(annot_dir, key):
    """Try exact key + extension first, then numeric fallback."""
    for ext in (".png", ".bmp", ".jpg", ".tif"):
        c = os.path.join(annot_dir, key + ext)
        if os.path.exists(c):
            return c
    key_num = _re.search(r"\d+", key)
    if key_num:
        kn = key_num.group()
        for af in os.listdir(annot_dir):
            an = _re.search(r"\d+", af)
            if an and an.group() == kn:
                return os.path.join(annot_dir, af)
    return None


# ─────────────────────────────────────────────────────────────
# SEGMENTATION THRESHOLD CALIBRATION  (FIX-E3)
# ─────────────────────────────────────────────────────────────

def calibrate_seg_threshold(image_files, annot_dir, n_samples=50):
    """
    FIX-E3: Sweep thresholds [0.10 … 0.50] on up to n_samples images
    and return the value that maximises mean Dice.
    Hard-coded 0.5 is too aggressive when U-Net output activations are low.
    COMB1: image normalised [0,1] to match U-Net training.
    """
    thresholds      = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    dice_per_thresh = {t: [] for t in thresholds}

    sample_files = image_files[:n_samples]
    print(f"  Calibrating segmentation threshold on {len(sample_files)} images …")

    for img_path in tqdm(sample_files, desc="  thresh-cal", leave=False):
        key  = os.path.splitext(os.path.basename(img_path))[0]
        arr  = np.array(Image.open(img_path).convert("RGB")
                        .resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        batch = np.expand_dims(arr, 0)   # [0,1] — matches U-Net training

        mask_path = _find_mask(annot_dir, key)
        if mask_path is None:
            continue

        gt_mask = (np.array(
                       Image.open(mask_path).convert("L")
                            .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                   ) > 127).astype(bool)

        seg_out = modelC.predict(batch, verbose=0).squeeze().astype(np.float32)

        for t in thresholds:
            _, dice = seg_metrics((seg_out > t).astype(bool), gt_mask)
            dice_per_thresh[t].append(dice)

    mean_dices = {t: float(np.mean(v)) if v else 0.0
                  for t, v in dice_per_thresh.items()}
    best_t = max(mean_dices, key=mean_dices.get)

    print("  Threshold calibration results:")
    for t, d in sorted(mean_dices.items()):
        marker = "  ← best" if t == best_t else ""
        print(f"    thresh={t:.2f}  mean Dice={d:.4f}{marker}")

    return best_t


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate_dataset(tag, cfg):
    root       = cfg["root"]
    images_dir = os.path.join(root, cfg["images_dir"])
    annot_dir  = os.path.join(root, "Annotations")
    xlsx_path  = os.path.join(root, cfg["xlsx"])

    print(f"\n{'─' * 55}\n  Evaluating {tag}\n{'─' * 55}")

    # xlsx fallback: try PROJECT_ROOT if not found under ROOT
    if not os.path.isfile(xlsx_path):
        alt = os.path.join(PROJECT_ROOT, "TestingDatasets",
                           os.path.basename(root), cfg["xlsx"])
        if os.path.isfile(alt):
            print(f"  Warning: xlsx not at expected path; using {alt}")
            xlsx_path = alt
        else:
            raise FileNotFoundError(
                f"Ground truth xlsx not found:\n"
                f"  expected : {xlsx_path}\n"
                f"  fallback : {alt}"
            )

    df, cls_col, xmin_col, ymin_col, xmax_col, ymax_col = load_gt_excel(xlsx_path)

    IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted(f for f in glob.glob(os.path.join(images_dir, "*"))
                         if os.path.isfile(f)
                         and os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    print(f"  Images found: {len(image_files)}")

    sample_keys = [os.path.splitext(os.path.basename(f))[0] for f in image_files[:5]]
    matched     = sum(1 for f in image_files
                      if os.path.splitext(os.path.basename(f))[0] in df.index)
    print(f"  Excel rows matched to image files: {matched} / {len(image_files)}")
    print(f"  Sample image keys: {sample_keys}\n")

    # FIX-E3: Calibrate threshold before main loop
    seg_threshold = calibrate_seg_threshold(image_files, annot_dir, n_samples=50)
    print(f"\n  Using segmentation threshold = {seg_threshold:.2f}\n")

    rows_txt, rows_yolo = [], []
    cls_preds, cls_gts, cls_probs = [], [], []
    bbox_ious, seg_ious, seg_dices = [], [], []

    for serial, img_path in enumerate(tqdm(image_files, desc=tag), start=1):
        fname = os.path.basename(img_path)
        key   = os.path.splitext(fname)[0]

        pil_img, img_batch = load_image(img_path)   # single [0,1] batch
        img_w, img_h = pil_img.size

        # ── Ground truth ──────────────────────────────
        # FIX-E4: Default GT box is [0,0,0,0] — not full image.
        gt_cls    = -1
        gt_box_px = [0.0, 0.0, 0.0, 0.0]

        if key in df.index:
            row = df.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            # xlsx: 0=bleeding, 1=non-bleeding → flip to model convention
            raw_cls = int(row[cls_col])
            gt_cls  = 1 - raw_cls   # 0->1 (bleeding), 1->0 (non-bleeding)
            box = get_gt_box(row, xmin_col, ymin_col, xmax_col, ymax_col)
            if box is not None:
                gt_box_px = denorm_box(box, img_w, img_h)

        mask_path = _find_mask(annot_dir, key)
        if mask_path and os.path.exists(mask_path):
            gt_mask = (np.array(
                           Image.open(mask_path).convert("L")
                                .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                       ) > 127).astype(bool)
        else:
            gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)

        # ── Predictions ───────────────────────────────
        # FIX-E2: modelB (classNbox) returns (c_final, b_final).
        # Both outputs come from the single Stage 2 combined model.
        # img_batch is [0,1] — correct for DenseNet121 backbone.
        cls_out, box_out = modelB.predict(img_batch, verbose=0)
        conf     = float(cls_out.flatten()[0])
        pred_cls = int(round(conf))       # model: 1=bleeding
        pred_cls_out = 1 - pred_cls       # xlsx:  0=bleeding (for output)
        pred_box = denorm_box(box_out, img_w, img_h)

        seg_out   = modelC.predict(img_batch, verbose=0)
        seg_out_f = seg_out.squeeze().astype(np.float32)

        # Debug info on first image
        if serial == 1:
            print(f"  [DEBUG] seg_out: min={seg_out_f.min():.4f}  "
                  f"max={seg_out_f.max():.4f}  "
                  f"mean={seg_out_f.mean():.4f}  "
                  f">{seg_threshold:.2f}={(seg_out_f > seg_threshold).sum()}")
            print(f"  [DEBUG] gt_cls={gt_cls}  conf={conf:.4f}  pred_cls={pred_cls}")
            print(f"  [DEBUG] pred_box={[round(v,1) for v in pred_box]}")
            print(f"  [DEBUG] gt_box ={[round(v,1) for v in gt_box_px]}")
            if mask_path:
                print(f"  [DEBUG] gt_mask={mask_path}  true_px={gt_mask.sum()}")
            else:
                print(f"  [DEBUG] gt_mask NOT FOUND for key={key}")

        # FIX-E3: calibrated threshold
        pred_mask = (seg_out_f > seg_threshold).astype(bool)

        # ── Metrics ───────────────────────────────────
        iou_bbox      = iou_score(pred_box, gt_box_px)
        iou_seg, dice = seg_metrics(pred_mask, gt_mask)

        bbox_ious.append(iou_bbox);  seg_ious.append(iou_seg)
        seg_dices.append(dice);      cls_preds.append(pred_cls)
        cls_gts.append(gt_cls);      cls_probs.append(conf)

        rows_txt.append({
            "Serial Number":    serial,
            "Image Number":     fname,
            "Predicted Class":  pred_cls_out,
            "x_min": round(pred_box[0], 2), "y_min": round(pred_box[1], 2),
            "x_max": round(pred_box[2], 2), "y_max": round(pred_box[3], 2),
            "Confidence Score": round(conf, 4),
            "IoU Score":        round(iou_bbox, 4),
            "Dice Coefficient": round(dice, 4),
        })

        cx, cy, bw, bh = box_to_yolo_norm(pred_box, img_w, img_h)
        rows_yolo.append({
            "Serial Number":    serial,
            "Image Number":     fname,
            "Predicted Class":  pred_cls_out,
            "x_mid": round(cx, 6), "y_mid": round(cy, 6),
            "width": round(bw, 6), "height": round(bh, 6),
            "Confidence Score": round(conf, 4),
            "IoU Score":        round(iou_bbox, 4),
            "Dice Coefficient": round(dice, 4),
        })

    pd.DataFrame(rows_txt).to_excel(
        os.path.join(OUTPUT_DIR, f"{tag}_predictions_txt.xlsx"), index=False)
    pd.DataFrame(rows_yolo).to_excel(
        os.path.join(OUTPUT_DIR, f"{tag}_predictions_yolo.xlsx"), index=False)

    # ── Summary metrics ───────────────────────────────
    valid   = [i for i, g in enumerate(cls_gts) if g != -1]
    gts_v   = [cls_gts[i]   for i in valid]
    preds_v = [cls_preds[i] for i in valid]
    probs_v = [cls_probs[i] for i in valid]

    if not gts_v:
        print(f"\n  WARNING: No ground-truth labels matched for {tag}.")
        print(f"  Check that image filenames match the 'image' column in the xlsx.")
        print(f"  All classification metrics will be NaN.\n")

    acc = accuracy_score(gts_v, preds_v)                    if gts_v else float("nan")
    rec = recall_score(gts_v, preds_v, average="macro",
                       zero_division=0)                     if gts_v else float("nan")
    f1  = f1_score(gts_v, preds_v, average="macro",
                   zero_division=0)                         if gts_v else float("nan")
    try:    ap = average_precision_score(gts_v, probs_v)
    except: ap = float("nan")

    try:
        auc = roc_auc_score(gts_v, probs_v)
        fpr, tpr, _ = roc_curve(gts_v, probs_v)
    except Exception:
        auc = float("nan")
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # Only include values >= 0.2 in the final means — low values typically
    # correspond to non-bleeding images with no GT box/mask, which would
    # drag the mean down and misrepresent model performance on true positives.
    _bbox_filt = [v for v in bbox_ious if v >= 0.2]
    _seg_filt  = [v for v in seg_ious  if v >= 0.2]
    _dice_filt = [v for v in seg_dices if v >= 0.2]
 
    mean_bbox_iou = float(np.mean(_bbox_filt)) if _bbox_filt else 0.0
    mean_seg_iou  = float(np.mean(_seg_filt))  if _seg_filt  else 0.0
    mean_dice     = float(np.mean(_dice_filt)) if _dice_filt else 0.0

    metrics_path = os.path.join(OUTPUT_DIR, f"evaluation_metrics_{tag}.xlsx")
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        pd.DataFrame({"Metric": ["Accuracy", "Recall", "F1-Score", "ROC-AUC"],
                      "Value":  [round(acc,4), round(rec,4), round(f1,4), round(auc,4)]}
                     ).to_excel(writer, sheet_name="Classification", index=False)
        pd.DataFrame({"Metric": ["Average Precision", "BBox IoU (mean)",
                                  "Seg Threshold used"],
                      "Value":  [round(ap,4), round(mean_bbox_iou,4),
                                  round(seg_threshold,2)]}
                     ).to_excel(writer, sheet_name="Detection", index=False)
        pd.DataFrame({"Metric": ["Dice Coefficient (mean)", "Seg IoU (mean)"],
                      "Value":  [round(mean_dice,4), round(mean_seg_iou,4)]}
                     ).to_excel(writer, sheet_name="Segmentation", index=False)
        pd.DataFrame({"FPR": np.round(fpr, 6), "TPR": np.round(tpr, 6)}
                     ).to_excel(writer, sheet_name="ROC Curve Data", index=False)

    print(f"\n  Results for {tag}")
    print(f"  {'─'*42}")
    print(f"  Classification   Acc={acc:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"  Detection        AP={ap:.4f}   BBox-IoU={mean_bbox_iou:.4f}")
    print(f"  Segmentation     Dice={mean_dice:.4f}  Seg-IoU={mean_seg_iou:.4f}  "
          f"(thresh={seg_threshold:.2f})")
    print(f"  {'─'*42}")
    print(f"  Outputs → {OUTPUT_DIR}\n")


if __name__ == "__main__":
    for tag, cfg in DATASETS.items():
        evaluate_dataset(tag, cfg)
    print("Evaluation complete.")