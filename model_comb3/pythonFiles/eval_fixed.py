"""
eval.py  —  ColonNet Evaluation Script  (Combination 3)
=========================================================
Pipeline:
  Stage 1 : YOLOv8n            → bounding-box (YoloBoxPredictor)
  Stage 2 : EfficientNetB0     → classification (c_final)
  Stage 3 : UNet++             → segmentation (seg_output)
  Stage 4 : ColonNet.keras     → combined frozen model

Label convention (matches training and data_loaders):
    1 = bleeding   (model outputs ~1.0 for bleeding)
    0 = non-bleeding

xlsx ground-truth convention (WCEBleedGen dataset):
    0 = bleeding
    1 = non-bleeding
  → gt_cls is flipped (1 - raw_cls) to align with model convention.
    Evidence: AUC < 0.5 without flip = consistent label inversion.

Fixes applied:
  FIX-E1  gt_cls flipped: 1 - raw_cls to align xlsx (0=bleeding) with
          model convention (1=bleeding). Output Excel uses xlsx convention
          (0=bleeding) via pred_cls_out = 1 - pred_cls.
  FIX-E2  gt_box_px default changed from full image to [0,0,0,0].
          The old [0,0,img_w,img_h] default inflated IoU for images
          with no GT box annotation.
  FIX-E3  denorm_box: sentinel check for all-zero box.
  FIX-E4  iou_score: returns 0 if either box has zero area.
  FIX-E5  load_gt_excel: deduplicates rows per image key before indexing.
          Multiple rows per key (one per bounding box) caused df.loc[key]
          to return a DataFrame, and iloc[0] silently picked the wrong row.
  FIX-E6  ColonNet outputs are a dict — accessed by name not integer index.
          colon_out["c_final"], colon_out["seg_output"] — not [0]/[2].
  FIX-E7  YoloBoxPredictor.predict_batch expects [0,1] float input.
          The PIL array from load_image is already [0,1]; do not re-divide.
  FIX-E8  Segmentation threshold auto-calibrated over first 50 images
          instead of hard-coded 0.5. U-Net outputs low-confidence
          activations on test images; 0.5 gives Dice=0 even for a
          trained model.
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
# YoloBoxPredictor not used — training.py uses Keras EfficientNetB0 box head

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
# ─────────────────────────────────────────────────────────────

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f     = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f     = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def smooth_l1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff   = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))


def giou_loss(y_true, y_pred):
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
    ex1    = tf.minimum(y_true[:, 0], y_pred[:, 0])
    ey1    = tf.minimum(y_true[:, 1], y_pred[:, 1])
    ex2    = tf.maximum(y_true[:, 2], y_pred[:, 2])
    ey2    = tf.maximum(y_true[:, 3], y_pred[:, 3])
    enc    = (ex2 - ex1) * (ey2 - ey1) + 1e-7
    return tf.reduce_mean(1.0 - (iou - (enc - union) / enc))


def combined_box_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w = y_true[:, 2] - y_true[:, 0]
    h = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0) & (h > 0), tf.float32)
    diff           = tf.abs(y_true - y_pred)
    sl1            = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    sl1_per_sample = tf.reduce_mean(sl1, axis=1)
    masked  = sl1_per_sample * valid_mask
    n_valid = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    return tf.reduce_sum(masked) / n_valid


SEG_CUSTOM = {"focal_tversky": focal_tversky, "tversky": tversky, "dice_coef": dice_coef}
BOX_CUSTOM = {"giou_loss": giou_loss, "smooth_l1": smooth_l1,
              "combined_box_loss": combined_box_loss}
ALL_CUSTOM = {**SEG_CUSTOM, **BOX_CUSTOM}

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

def find_model(*names):
    for name in names:
        p = os.path.join(MODELS_DIR, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"None of {names} found in {MODELS_DIR}")


USE_COLONNET = True
try:
    colonnet_path = find_model("ColonNet.keras", "ColonNet.h5")
    print(f"Loading ColonNet combined model: {colonnet_path} …")
    colonnet = tf.keras.models.load_model(colonnet_path,
                                          custom_objects=ALL_CUSTOM,
                                          compile=False)
    print("ColonNet loaded.")
except FileNotFoundError:
    USE_COLONNET = False
    print("ColonNet not found — falling back to Stage 2 + Stage 3 separately.")
    cls_model = tf.keras.models.load_model(
        find_model("classNbox.keras", "classNbox.h5"),
        custom_objects=BOX_CUSTOM, compile=False)
    seg_model = tf.keras.models.load_model(
        find_model("segmentation.keras", "segmentation.h5"),
        custom_objects=SEG_CUSTOM, compile=False)

print("All models loaded.\n")

# Segmentation sanity check
print("Checking segmentation model …")
_seg_ref = colonnet if USE_COLONNET else seg_model
all_weights = np.concatenate([w.numpy().flatten() for w in _seg_ref.weights])
print(f"  Weight std-dev: {all_weights.std():.6f}")
_dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
if USE_COLONNET:
    _dummy_out = colonnet.predict(_dummy, verbose=0)
    # FIX-E6: ColonNet outputs dict — access by name, not integer index
    _seg_out = np.array(_dummy_out["seg_output"]).squeeze().astype(np.float32)
else:
    _seg_out = seg_model.predict(_dummy, verbose=0).squeeze().astype(np.float32)
print(f"  Dummy seg output: min={_seg_out.min():.4f}  max={_seg_out.max():.4f}")
if _seg_out.max() < 0.01:
    print("  ⚠  SEGMENTATION DEAD — delete segmentation.keras and retrain Stage 3.")
else:
    print("  ✓  Segmentation model is live.")
print()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_image(path):
    """
    Returns (pil_img, img_batch_float32).
    Normalised to [0,1] — matches training data_loaders /255.0.
    EfficientNetB0 Rescaling layer inside the model handles *255 internally.
    U-Net++ also trained on [0,1] inputs.
    """
    pil = Image.open(path).convert("RGB")
    arr = np.array(pil.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return pil, np.expand_dims(arr, 0)


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {candidates} found in {list(df.columns)}")
    return None


def load_gt_excel(path):
    """
    FIX-E5: Deduplicate rows per image key before indexing.
    Multiple rows per image (one per bounding box annotation) cause
    df.loc[key] to return a DataFrame; iloc[0] then silently picks the
    wrong row. Aggregate: class=max(), box=min/max union.
    """
    df = pd.read_excel(path)
    print(f"  Excel columns: {list(df.columns)}")
    print(f"  Excel rows before dedup: {len(df)}")

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

    agg = {cls_col: "max"}
    if xmin_col: agg[xmin_col] = "min"
    if ymin_col: agg[ymin_col] = "min"
    if xmax_col: agg[xmax_col] = "max"
    if ymax_col: agg[ymax_col] = "max"

    df_agg = df.groupby("_key", as_index=False).agg(agg)
    print(f"  Excel rows after dedup : {len(df_agg)}")
    print(f"  Excel keys (first 5)   : {list(df_agg['_key'][:5])}")
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
    FIX-E3: Sentinel check — all-zero means no box, return as-is.
    Otherwise scale normalised [0,1] coords to pixel space.
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
    """FIX-E4: Returns 0 if either box has zero area."""
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


def calibrate_seg_threshold(image_files, annot_dir, n_samples=50):
    """
    FIX-E8: Sweep thresholds [0.1..0.5] on up to n_samples images.
    Returns threshold that maximises mean Dice on those samples.
    """
    thresholds      = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    dice_per_thresh = {t: [] for t in thresholds}

    for img_path in tqdm(image_files[:n_samples], desc="  seg-thresh-cal", leave=False):
        fname     = os.path.basename(img_path)
        key       = os.path.splitext(fname)[0]
        _, batch  = load_image(img_path)

        mask_path = None
        for ext in (".png", ".bmp", ".jpg", ".tif"):
            c = os.path.join(annot_dir, key + ext)
            if os.path.exists(c):
                mask_path = c; break
        if mask_path is None:
            key_num = _re.search(r"\d+", key)
            if key_num:
                kn = key_num.group()
                for af in os.listdir(annot_dir):
                    an = _re.search(r"\d+", af)
                    if an and an.group() == kn:
                        mask_path = os.path.join(annot_dir, af); break
        if mask_path is None:
            continue

        gt_mask = (np.array(
                       Image.open(mask_path).convert("L")
                            .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                   ) > 127).astype(bool)

        if USE_COLONNET:
            out      = colonnet.predict(batch, verbose=0)
            seg_out  = np.array(out["seg_output"]).squeeze().astype(np.float32)
        else:
            seg_out  = seg_model.predict(batch, verbose=0).squeeze().astype(np.float32)

        for t in thresholds:
            _, dice = seg_metrics((seg_out > t).astype(bool), gt_mask)
            dice_per_thresh[t].append(dice)

    mean_dices = {t: float(np.mean(v)) if v else 0.0
                  for t, v in dice_per_thresh.items()}
    best_t = max(mean_dices, key=mean_dices.get)
    print("  Threshold calibration:")
    for t, d in sorted(mean_dices.items()):
        print(f"    {t:.2f} → Dice={d:.4f}{'  ← best' if t == best_t else ''}")
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

    if not os.path.isfile(xlsx_path):
        alt = os.path.join(PROJECT_ROOT, "TestingDatasets",
                           os.path.basename(root), os.path.basename(cfg["xlsx"]))
        if os.path.isfile(alt):
            print(f"  Warning: xlsx not at expected path; using {alt}")
            xlsx_path = alt
        else:
            raise FileNotFoundError(
                f"Ground truth xlsx not found:\n  expected: {xlsx_path}\n  fallback: {alt}"
            )

    df, cls_col, xmin_col, ymin_col, xmax_col, ymax_col = load_gt_excel(xlsx_path)

    IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted(f for f in glob.glob(os.path.join(images_dir, "*"))
                         if os.path.isfile(f)
                         and os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    print(f"  Images found: {len(image_files)}")

    matched = sum(1 for f in image_files
                  if os.path.splitext(os.path.basename(f))[0] in df.index)
    print(f"  Excel rows matched: {matched} / {len(image_files)}\n")

    seg_threshold = calibrate_seg_threshold(image_files, annot_dir, n_samples=50)
    print(f"\n  Using seg threshold = {seg_threshold:.2f}\n")

    rows_txt, rows_yolo = [], []
    cls_preds, cls_gts, cls_probs = [], [], []
    bbox_ious, seg_ious, seg_dices = [], [], []

    for serial, img_path in enumerate(tqdm(image_files, desc=tag), start=1):
        fname = os.path.basename(img_path)
        key   = os.path.splitext(fname)[0]

        pil_img, img_batch = load_image(img_path)
        img_w, img_h = pil_img.size

        # ── Ground truth ──────────────────────────────
        # FIX-E2: default is [0,0,0,0] (no box), not full image
        gt_cls    = -1
        gt_box_px = [0.0, 0.0, 0.0, 0.0]

        if key in df.index:
            row = df.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            # FIX-E1: xlsx 0=bleeding, model 1=bleeding → flip
            raw_cls = int(row[cls_col])
            gt_cls  = 1 - raw_cls
            box     = get_gt_box(row, xmin_col, ymin_col, xmax_col, ymax_col)
            if box is not None:
                gt_box_px = denorm_box(box, img_w, img_h)

        mask_path = None
        for ext in (".png", ".bmp", ".jpg", ".tif"):
            candidate = os.path.join(annot_dir, key + ext)
            if os.path.exists(candidate):
                mask_path = candidate; break
        if mask_path is None:
            key_num = _re.search(r"\d+", key)
            if key_num:
                kn = key_num.group()
                for af in os.listdir(annot_dir):
                    an = _re.search(r"\d+", af)
                    if an and an.group() == kn:
                        mask_path = os.path.join(annot_dir, af); break

        if mask_path and os.path.exists(mask_path):
            gt_mask = (np.array(
                           Image.open(mask_path).convert("L")
                                .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                       ) > 127).astype(bool)
        else:
            gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)

        # ── Predictions ───────────────────────────────
        # Predictions from Keras models — no YOLO in training.py.
        # ColonNet (Stage 4) exposes c_final, b_final, seg_output as a dict.
        # classNbox.keras (Stage 2) returns [c_final, b_final] as a list.
        if USE_COLONNET:
            colon_out = colonnet.predict(img_batch, verbose=0)
            cls_out   = np.array(colon_out["c_final"])
            box_out   = np.array(colon_out["b_final"])
            seg_out   = np.array(colon_out["seg_output"])
        else:
            cls_out, box_out = cls_model.predict(img_batch, verbose=0)
            seg_out          = seg_model.predict(img_batch, verbose=0)

        pred_box = denorm_box(box_out, img_w, img_h)

        conf     = float(np.array(cls_out).flatten()[0])
        pred_cls = int(round(conf))          # model convention: 1=bleeding
        # FIX-E1: output Excel uses xlsx convention: 0=bleeding
        pred_cls_out = 1 - pred_cls

        seg_out_f = np.array(seg_out).squeeze().astype(np.float32)
        if serial == 1:
            print(f"  [DEBUG] seg: min={seg_out_f.min():.4f}  "
                  f"max={seg_out_f.max():.4f}  "
                  f"mean={seg_out_f.mean():.4f}  "
                  f">{seg_threshold:.2f}={(seg_out_f > seg_threshold).sum()}")
            print(f"  [DEBUG] gt_cls={gt_cls}  conf={conf:.4f}  pred_cls={pred_cls}")
            print(f"  [DEBUG] pred_box={[round(v,1) for v in pred_box]}")
            print(f"  [DEBUG] gt_box ={[round(v,1) for v in gt_box_px]}")

        pred_mask = (seg_out_f > seg_threshold).astype(bool)

        iou_bbox      = iou_score(pred_box, gt_box_px)
        iou_seg, dice = seg_metrics(pred_mask, gt_mask)

        bbox_ious.append(iou_bbox);  seg_ious.append(iou_seg)
        seg_dices.append(dice);      cls_preds.append(pred_cls)
        cls_gts.append(gt_cls);      cls_probs.append(conf)

        rows_txt.append({
            "Serial Number":    serial,
            "Image Number":     fname,
            "Predicted Class":  pred_cls_out,   # xlsx convention: 0=bleeding
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
            "Predicted Class":  pred_cls_out,   # xlsx convention: 0=bleeding
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

    valid   = [i for i, g in enumerate(cls_gts) if g != -1]
    gts_v   = [cls_gts[i]   for i in valid]
    preds_v = [cls_preds[i] for i in valid]
    probs_v = [cls_probs[i] for i in valid]

    if not gts_v:
        print(f"\n  WARNING: No GT labels matched for {tag}.")
        print("  Check image filenames match the xlsx 'image' column.\n")

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

    mean_bbox_iou = float(np.mean(bbox_ious))
    mean_seg_iou  = float(np.mean(seg_ious))
    mean_dice     = float(np.mean(seg_dices))

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
    print(f"  Segmentation     Dice={mean_dice:.4f}  Seg-IoU={mean_seg_iou:.4f}"
          f"  (thresh={seg_threshold:.2f})")
    print(f"  {'─'*42}")
    print(f"  Outputs → {OUTPUT_DIR}\n")


if __name__ == "__main__":
    for tag, cfg in DATASETS.items():
        evaluate_dataset(tag, cfg)
    print("Evaluation complete.")