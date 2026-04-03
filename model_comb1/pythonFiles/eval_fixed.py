"""
eval.py  —  ColonNet Evaluation Script
========================================
Test dataset structure:
  TestingDatasets/
    Test Dataset 1/
      Annotations/          ← binary segmentation masks
      Unmarked Images/      ← input images
      Test Dataset 1 TXT (True labels).xlsx

    Test Dataset 2/
      Annotations/
      Images/
      Test Dataset 2 TXT (True labels).xlsx

Label convention (matches training):
    0 = bleeding
    1 = non-bleeding
"""

import os
import glob
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              average_precision_score,
                              roc_auc_score, roc_curve)

from utils.losses import focal_tversky, tversky

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TESTING_ROOT = os.path.join(ROOT, "TestingDatasets")   # fixed: no extra subfolder
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
# CUSTOM OBJECTS  (must match training definitions exactly)
# ─────────────────────────────────────────────────────────────

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f     = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f     = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


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


SEG_CUSTOM = {"focal_tversky": focal_tversky, "tversky": tversky, "dice_coef": dice_coef}
BOX_CUSTOM = {"giou_loss": giou_loss}

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

def find_model(*names):
    for name in names:
        p = os.path.join(MODELS_DIR, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"None of {names} found in {MODELS_DIR}")


print("Loading bbox model …")
modelA = tf.keras.models.load_model(
    find_model("CheckPoint1.h5", "CheckPoint1.keras"),
    custom_objects=BOX_CUSTOM, compile=False)

print("Loading classification model …")
modelB = tf.keras.models.load_model(
    find_model("classNbox.h5", "classNbox.keras"),
    custom_objects=BOX_CUSTOM, compile=False)

print("Loading segmentation model …")
modelC = tf.keras.models.load_model(
    find_model("segmentation.h5", "segmentation.keras"),
    custom_objects=SEG_CUSTOM, compile=False)

print("All models loaded.\n")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_image(path):
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
    df      = pd.read_excel(path)
    img_col = find_col(df, ["image_name", "image", "filename", "file",
                             "image_id", "img", "name", "Image", "Filename"])
    cls_col = find_col(df, ["class_label", "class", "label", "labels",
                             "bleeding", "annotation", "y", "target", "Class"])
    xmin_col = find_col(df, ["x_min", "xmin", "X_min", "Xmin"], required=False)
    ymin_col = find_col(df, ["y_min", "ymin", "Y_min", "Ymin"], required=False)
    xmax_col = find_col(df, ["x_max", "xmax", "X_max", "Xmax"], required=False)
    ymax_col = find_col(df, ["y_max", "ymax", "Y_max", "Ymax"], required=False)

    df["_key"] = (df[img_col].astype(str)
                  .apply(lambda x: os.path.splitext(os.path.basename(x))[0]))
    return df.set_index("_key"), cls_col, xmin_col, ymin_col, xmax_col, ymax_col


def get_gt_box(row, xmin_col, ymin_col, xmax_col, ymax_col):
    if any(c is None for c in [xmin_col, ymin_col, xmax_col, ymax_col]):
        return None
    try:
        return [float(row[xmin_col]), float(row[ymin_col]),
                float(row[xmax_col]), float(row[ymax_col])]
    except Exception:
        return None


def denorm_box(raw, img_w, img_h):
    b = np.array(raw).flatten()[:4].astype(float)
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
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (aA + aB - inter + 1e-7)


def seg_metrics(pred_mask, gt_mask):
    p     = pred_mask.flatten().astype(bool)
    g     = gt_mask.flatten().astype(bool)
    inter = np.logical_and(p, g).sum()
    iou_v = inter / (np.logical_or(p, g).sum() + 1e-7)
    dice  = (2 * inter) / (p.sum() + g.sum() + 1e-7)
    return float(iou_v), float(dice)


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate_dataset(tag, cfg):
    root       = cfg["root"]
    images_dir = os.path.join(root, cfg["images_dir"])
    annot_dir  = os.path.join(root, "Annotations")
    xlsx_path  = os.path.join(root, cfg["xlsx"])

    print(f"\n{'─' * 55}\n  Evaluating {tag}\n{'─' * 55}")

    df, cls_col, xmin_col, ymin_col, xmax_col, ymax_col = load_gt_excel(xlsx_path)

    IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted(f for f in glob.glob(os.path.join(images_dir, "*"))
                         if os.path.isfile(f)
                         and os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    print(f"  Images found: {len(image_files)}\n")

    rows_txt, rows_yolo = [], []
    cls_preds, cls_gts, cls_probs = [], [], []
    bbox_ious, seg_ious, seg_dices = [], [], []

    for serial, img_path in enumerate(tqdm(image_files, desc=tag), start=1):
        fname = os.path.basename(img_path)
        key   = os.path.splitext(fname)[0]

        pil_img, img_batch = load_image(img_path)
        img_w, img_h       = pil_img.size

        # ── Ground truth ─────────────────────────────
        gt_cls    = -1
        gt_box_px = [0, 0, img_w, img_h]

        if key in df.index:
            row = df.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            gt_cls = int(row[cls_col])
            box    = get_gt_box(row, xmin_col, ymin_col, xmax_col, ymax_col)
            if box is not None:
                gt_box_px = box

        mask_path = os.path.join(annot_dir, key + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(annot_dir, key + ".bmp")
        gt_mask = (np.array(Image.open(mask_path).resize((IMG_SIZE, IMG_SIZE))) > 127
                   ).astype(bool) if os.path.exists(mask_path) \
                   else np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)

        # ── Predictions ──────────────────────────────
        cls_out, _  = modelB.predict(img_batch, verbose=0)
        conf        = float(cls_out.flatten()[0])
        pred_cls    = int(round(conf))          # 0=bleeding, 1=non-bleeding

        _, box_out  = modelA.predict(img_batch, verbose=0)
        pred_box    = denorm_box(box_out, img_w, img_h)

        seg_out     = modelC.predict(img_batch, verbose=0)
        pred_mask   = (seg_out.squeeze() > 0.1).astype(bool)

        # ── Metrics ──────────────────────────────────
        iou_bbox      = iou_score(pred_box, gt_box_px)
        iou_seg, dice = seg_metrics(pred_mask, gt_mask)

        bbox_ious.append(iou_bbox);  seg_ious.append(iou_seg)
        seg_dices.append(dice);      cls_preds.append(pred_cls)
        cls_gts.append(gt_cls);      cls_probs.append(conf)

        rows_txt.append({
            "Serial Number": serial, "Image Number": fname,
            "Predicted Class": pred_cls,
            "x_min": round(pred_box[0], 2), "y_min": round(pred_box[1], 2),
            "x_max": round(pred_box[2], 2), "y_max": round(pred_box[3], 2),
            "Confidence Score": round(conf, 4),
            "IoU Score": round(iou_bbox, 4), "Dice Coefficient": round(dice, 4),
        })

        cx, cy, bw, bh = box_to_yolo_norm(pred_box, img_w, img_h)
        rows_yolo.append({
            "Serial Number": serial, "Image Number": fname,
            "Predicted Class": pred_cls,
            "x_mid": round(cx, 6), "y_mid": round(cy, 6),
            "width": round(bw, 6), "height": round(bh, 6),
            "Confidence Score": round(conf, 4),
            "IoU Score": round(iou_bbox, 4), "Dice Coefficient": round(dice, 4),
        })

    pd.DataFrame(rows_txt).to_excel(
        os.path.join(OUTPUT_DIR, f"{tag}_predictions_txt.xlsx"), index=False)
    pd.DataFrame(rows_yolo).to_excel(
        os.path.join(OUTPUT_DIR, f"{tag}_predictions_yolo.xlsx"), index=False)

    valid   = [i for i, g in enumerate(cls_gts) if g != -1]
    gts_v   = [cls_gts[i]   for i in valid]
    preds_v = [cls_preds[i] for i in valid]
    probs_v = [cls_probs[i] for i in valid]

    acc = accuracy_score(gts_v, preds_v)                       if gts_v else float("nan")
    rec = recall_score(gts_v, preds_v, average="macro",
                       zero_division=0)                        if gts_v else float("nan")
    f1  = f1_score(gts_v, preds_v, average="macro",
                   zero_division=0)                            if gts_v else float("nan")
    try:    ap = average_precision_score(gts_v, probs_v)
    except: ap = float("nan")

    # ── ROC-AUC ──────────────────────────────────────
    # probs_v is raw sigmoid output (~0=bleeding, ~1=non-bleeding).
    # roc_auc_score expects probability of the positive class (1=non-bleeding).
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
        pd.DataFrame({"Metric": ["Average Precision", "BBox IoU (mean)"],
                      "Value":  [round(ap,4), round(mean_bbox_iou,4)]}
                     ).to_excel(writer, sheet_name="Detection", index=False)
        pd.DataFrame({"Metric": ["Dice Coefficient (mean)", "Seg IoU (mean)"],
                      "Value":  [round(mean_dice,4), round(mean_seg_iou,4)]}
                     ).to_excel(writer, sheet_name="Segmentation", index=False)
        # ROC curve data as its own sheet for reference
        pd.DataFrame({"FPR": np.round(fpr, 6), "TPR": np.round(tpr, 6)}
                     ).to_excel(writer, sheet_name="ROC Curve Data", index=False)

    # ── Save ROC curve plot ───────────────────────────
    roc_path = os.path.join(OUTPUT_DIR, f"{tag}_roc_curve.png")
    fig, ax  = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {tag}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved → {roc_path}")

    print(f"\n  Results for {tag}")
    print(f"  {'─'*42}")
    print(f"  Classification   Acc={acc:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"  Detection        AP={ap:.4f}   BBox-IoU={mean_bbox_iou:.4f}")
    print(f"  Segmentation     Dice={mean_dice:.4f}  Seg-IoU={mean_seg_iou:.4f}")
    print(f"  {'─'*42}")
    print(f"  Outputs → {OUTPUT_DIR}\n")


if __name__ == "__main__":
    for tag, cfg in DATASETS.items():
        evaluate_dataset(tag, cfg)
    print("Evaluation complete.")