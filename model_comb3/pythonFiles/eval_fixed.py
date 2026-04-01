"""
eval.py  —  ColonNet Evaluation Script  (aligned with training.py Combination 3)
==================================================================================
Pipeline (matches training.py exactly):
  Stage 1 : YOLOv8n            → bounding-box regression   (YoloBoxPredictor)
  Stage 2 : EfficientNetB0     → classification             (c_final from ColonNet)
  Stage 3 : UNet++             → segmentation               (seg_output from ColonNet)
  Stage 4 : ColonNet.keras     → combined frozen model      (c_final, b_final, seg_output)

Model loading priority:
  Bounding box  → YoloBoxPredictor("SavedModels/yolo_best.pt")
  Classification + Segmentation → ColonNet.keras  (outputs: c_final, b_final, seg_output)
  Fallback      → classNbox.keras / segmentation.keras loaded separately

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

# ROOT = model_comb3 root, PROJECT_ROOT = repository root
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # model_comb3
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))         # repository root
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
from utils.yolo_box import YoloBoxPredictor   # matches training.py Stage 1 inference

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
# CUSTOM OBJECTS  (must match training.py definitions exactly)
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
    """Matches training.py giou_loss with valid-box masking."""
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
    giou   = 1.0 - (iou - (enc - union) / enc)
    valid_mask = tf.cast((area_t > 0), tf.float32)
    giou   = giou * valid_mask
    denom  = tf.reduce_sum(valid_mask) + 1e-7
    return tf.reduce_sum(giou) / denom


def combined_box_loss(y_true, y_pred):
    """GIoU + 0.5 * Smooth-L1, masked to valid boxes (area > 0). Matches training.py."""
    w = y_true[:, 2] - y_true[:, 0]
    h = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0) & (h > 0), tf.float32)
    loss = giou_loss(y_true, y_pred) + 0.5 * smooth_l1(y_true, y_pred)
    return tf.reduce_mean(loss * valid_mask)


SEG_CUSTOM = {"focal_tversky": focal_tversky, "tversky": tversky, "dice_coef": dice_coef}
BOX_CUSTOM = {"giou_loss": giou_loss, "smooth_l1": smooth_l1, "combined_box_loss": combined_box_loss}
ALL_CUSTOM = {**SEG_CUSTOM, **BOX_CUSTOM}

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# Matches training.py Stage 4 inference pattern exactly:
#   yolo  = YoloBoxPredictor("SavedModels/yolo_best.pt")
#   cls, _, masks = colon_net(images)
# ─────────────────────────────────────────────────────────────

def find_model(*names):
    for name in names:
        p = os.path.join(MODELS_DIR, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"None of {names} found in {MODELS_DIR}")


# Stage 1 — YOLOv8n bounding-box predictor (training.py: YoloBoxPredictor)
print("Loading YOLOv8n bbox predictor (Stage 1) …")
yolo_predictor = YoloBoxPredictor(find_model("yolo_best.pt"))

# Stage 4 — ColonNet combined model (outputs: c_final, b_final, seg_output)
# Falls back to loading Stage 2 + Stage 3 models separately if ColonNet is unavailable.
USE_COLONNET = True
try:
    colonnet_path = find_model("ColonNet.keras", "ColonNet.h5")
    print(f"Loading ColonNet combined model (Stage 4): {colonnet_path} …")
    colonnet = tf.keras.models.load_model(colonnet_path, custom_objects=ALL_CUSTOM,
                                          compile=False)
    print("ColonNet loaded — outputs: [c_final, b_final, seg_output]")
except FileNotFoundError:
    USE_COLONNET = False
    print("ColonNet not found — falling back to separate Stage 2 + Stage 3 models.")
    print("Loading classification model (Stage 2) …")
    cls_model = tf.keras.models.load_model(
        find_model("classNbox.keras", "classNbox.h5"),
        custom_objects=BOX_CUSTOM, compile=False)
    print("Loading segmentation model (Stage 3) …")
    seg_model = tf.keras.models.load_model(
        find_model("segmentation.keras", "segmentation.h5"),
        custom_objects=SEG_CUSTOM, compile=False)

print("All models loaded.\n")

# ── Segmentation model weight sanity check ──────────────────
print("Checking segmentation model weights …")
_seg_ref = colonnet if USE_COLONNET else seg_model
# For ColonNet, extract the UNet++ sub-model output by running a dummy forward pass
total_params  = _seg_ref.count_params()
nonzero_count = sum(int(np.count_nonzero(w.numpy())) for w in _seg_ref.weights)
all_weights   = np.concatenate([w.numpy().flatten() for w in _seg_ref.weights])
print(f"  Total params     : {total_params}")
print(f"  Non-zero weights : {nonzero_count} / {total_params}")
print(f"  Weight range     : min={all_weights.min():.6f}  max={all_weights.max():.6f}")
print(f"  Weight std-dev   : {all_weights.std():.6f}")

_dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
if USE_COLONNET:
    _dummy_out = colonnet.predict(_dummy, verbose=0)
    _seg_out   = _dummy_out[2].squeeze().astype(np.float32)  # seg_output is 3rd output
else:
    _seg_out = seg_model.predict(_dummy, verbose=0).squeeze().astype(np.float32)
print(f"  Dummy-input seg output: min={_seg_out.min():.6f}  max={_seg_out.max():.6f}"
      f"  mean={_seg_out.mean():.6f}")
if _seg_out.max() < 0.01:
    print("  ⚠  SEGMENTATION OUTPUT IS DEAD — model weights were never trained or zeroed.")
    print("     Delete segmentation.keras and re-run training.py Stage 3.")
else:
    print("  ✓  Segmentation model produces non-zero output.")
print()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_image(path):
    """Load and preprocess image to match training.py EfficientNetB0 input.
    EfficientNetB0 uses its own internal rescaling — pass raw [0,255] uint8
    via tf.keras.applications.efficientnet.preprocess_input, or as float32
    normalised to [0,1]. training.py uses load_data which applies /255.0."""
    pil = Image.open(path).convert("RGB")
    arr = np.array(pil.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    arr = arr / 255.0    # EfficientNetB0 preprocessing used in training.py
    return pil, np.expand_dims(arr, 0).astype(np.float32)


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
    """Convert normalised [x1,y1,x2,y2] → pixel coords."""
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

    if not os.path.isfile(xlsx_path):
        alt = os.path.join(PROJECT_ROOT, "TestingDatasets",
                           os.path.basename(root), os.path.basename(cfg["xlsx"]))
        if os.path.isfile(alt):
            print(f"  Warning: xlsx file not found at expected path; using {alt}")
            xlsx_path = alt
        else:
            raise FileNotFoundError(
                f"Ground truth xlsx not found:\n"
                f"  expected: {xlsx_path}\n"
                f"  fallback:  {alt}\n"
                f"Please verify the dataset path or file name."
            )

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

        mask_path = None
        for ext in (".png", ".bmp", ".jpg", ".tif"):
            candidate = os.path.join(annot_dir, key + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            import re as _re
            key_num = _re.search(r"\d+", key)
            if key_num:
                key_num = key_num.group()
                for af in os.listdir(annot_dir):
                    af_num = _re.search(r"\d+", af)
                    if af_num and af_num.group() == key_num:
                        mask_path = os.path.join(annot_dir, af)
                        break
        if mask_path and os.path.exists(mask_path):
            gt_mask = (np.array(
                           Image.open(mask_path).convert("L")
                                .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                       ) > 127).astype(bool)
        else:
            gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)

        # ── Predictions ──────────────────────────────
        # Stage 1: YOLOv8n bounding-box (training.py inference pattern)
        #   yolo.predict_batch expects a list of file paths or a numpy batch.
        #   Returns normalised [x1,y1,x2,y2] per image.
        yolo_boxes = yolo_predictor.predict_batch(
            np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)[None]
        )
        if yolo_boxes is not None and len(yolo_boxes) > 0:
            pred_box = denorm_box(yolo_boxes[0], img_w, img_h)
        else:
            # No detection — fall back to full image box
            pred_box = [0.0, 0.0, float(img_w), float(img_h)]

        # Stage 2 + 3: ColonNet (c_final=classification, b_final=box, seg_output=mask)
        # Training.py inference: cls, _, masks = colon_net(images)
        if USE_COLONNET:
            colon_out = colonnet.predict(img_batch, verbose=0)
            cls_out   = colon_out[0]   # c_final  — EfficientNetB0 sigmoid
            # colon_out[1] is b_final (EfficientNet box head); YOLO box used instead
            seg_out   = colon_out[2]   # seg_output — UNet++ sigmoid mask
        else:
            cls_out, _ = cls_model.predict(img_batch, verbose=0)
            seg_out    = seg_model.predict(img_batch.astype(np.float32), verbose=0)

        conf     = float(np.array(cls_out).flatten()[0])
        pred_cls = int(round(conf))    # 0=bleeding, 1=non-bleeding

        seg_out_f = np.array(seg_out).squeeze().astype(np.float32)
        if serial == 1:
            print(f"  [DEBUG] seg_out shape={seg_out_f.shape} dtype={seg_out_f.dtype} "
                  f"min={seg_out_f.min():.6f} max={seg_out_f.max():.6f} "
                  f"mean={seg_out_f.mean():.6f} >0.5={(seg_out_f > 0.5).sum()}")
            if mask_path:
                print(f"  [DEBUG] gt_mask={mask_path} true_px={gt_mask.sum()}")
            else:
                print(f"  [DEBUG] gt_mask NOT FOUND key={key}")
        pred_mask = (seg_out_f > 0.5).astype(bool)

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

    acc = accuracy_score(gts_v, preds_v)                        if gts_v else float("nan")
    rec = recall_score(gts_v, preds_v, average="macro",
                       zero_division=0)                         if gts_v else float("nan")
    f1  = f1_score(gts_v, preds_v, average="macro",
                   zero_division=0)                             if gts_v else float("nan")
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
                      "Value":  [round(acc, 4), round(rec, 4), round(f1, 4), round(auc, 4)]}
                     ).to_excel(writer, sheet_name="Classification", index=False)
        pd.DataFrame({"Metric": ["Average Precision", "BBox IoU (mean)"],
                      "Value":  [round(ap, 4), round(mean_bbox_iou, 4)]}
                     ).to_excel(writer, sheet_name="Detection", index=False)
        pd.DataFrame({"Metric": ["Dice Coefficient (mean)", "Seg IoU (mean)"],
                      "Value":  [round(mean_dice, 4), round(mean_seg_iou, 4)]}
                     ).to_excel(writer, sheet_name="Segmentation", index=False)
        pd.DataFrame({"FPR": np.round(fpr, 6), "TPR": np.round(tpr, 6)}
                     ).to_excel(writer, sheet_name="ROC Curve Data", index=False)

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