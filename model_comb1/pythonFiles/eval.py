import os
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from utils.losses import focal_tversky, tversky # Needed for loading segmentation model

# =============================================================
#                 MODELS (shared for both datasets)
# =============================================================
# Use TensorFlow to load Keras models
print("Loading CheckPoint1.h5...")
modelA = tf.keras.models.load_model("SavedModels\\CheckPoint1.h5", compile=False)
print("Loading classNbox.h5...")
modelB = tf.keras.models.load_model("SavedModels\\classNbox.h5", compile=False)
print("Loading segmentation.keras...")
# Custom objects are needed for custom loss functions used during training
modelC = tf.keras.models.load_model(
    "SavedModels\\segmentation.keras",
    custom_objects={"focal_tversky": focal_tversky, "tversky": tversky},
    compile=False
)

# =============================================================
#                 HELPERS
# =============================================================

def read_yolo_bbox(txt_file, img_w, img_h):
    line = open(txt_file, "r").read().strip().split()
    cls, cx, cy, w, h = map(float, line)
    x1 = img_w * (cx - w/2); y1 = img_h * (cy - h/2)
    x2 = img_w * (cx + w/2); y2 = img_h * (cy + h/2)
    return int(cls), [x1, y1, x2, y2]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-7)

def seg_metrics(pred_mask, gt_mask):
    pred = pred_mask.flatten(); gt = gt_mask.flatten()
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou_val = intersection / (union + 1e-7)
    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-7)
    return iou_val, dice

def _find_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"No matching column found. Tried: {candidates}. Available columns: {list(df.columns)}")
    return None

# =============================================================
#                 EVALUATOR FOR ANY DATASET
# =============================================================

def evaluate_dataset(DATASET_ROOT, dataset_name):

    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"             Evaluating {dataset_name}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # auto-detect image folder
    if os.path.exists(os.path.join(DATASET_ROOT, "Unmarked Images")):
        IMAGES_DIR = os.path.join(DATASET_ROOT, "Unmarked Images")
    else:
        IMAGES_DIR = os.path.join(DATASET_ROOT, "Images")

    LABELS_DIR = os.path.join(DATASET_ROOT, "Labels")
    MASKS_DIR = os.path.join(DATASET_ROOT, "Annotations")

    # excel file
    spreadsheet_path = r"C:\Users\Shubham\Desktop\ColonNet\Test Dataset for Auto-WCEBleedGen Challenge version 2\True Class labels only for both test dataset 1 and 2.xlsx"
    if not os.path.exists(spreadsheet_path):
        print(f"ERROR: Could not find the 'True Class labels only for both test dataset 1 and 2.xlsx' file in {DATASET_ROOT}")
        return
    
    # Load spreadsheet / csv that lists filenames and classes
    df = pd.read_excel(spreadsheet_path) if spreadsheet_path.lower().endswith(('.xls','.xlsx')) else pd.read_csv(spreadsheet_path)
    
    # Flexible column lookup (handles 'filename','file','image','image_id', etc.)
    file_col = _find_column(df, ['filename','file','image','image_id','img','name','Image'])
    class_col = _find_column(df, ['class','label','labels','bleeding','annotation','y','target'])
    
    # Build maps using detected columns
    class_map = dict(zip(df[file_col].astype(str), df[class_col]))

    bbox_ious = []
    cls_preds = []
    cls_gts = []
    seg_ious = []
    seg_dices = []

    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*")))

    for img_path in tqdm(image_files, desc=f"{dataset_name}"):
        fname = os.path.basename(img_path)
        name_no_ext = os.path.splitext(fname)[0]

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        # Resize and normalize image using NumPy/PIL, then add batch dimension
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # ----------- GT BBOX ----------
        label_file = os.path.join(LABELS_DIR, name_no_ext + ".txt")
        cls_gt, gt_box = read_yolo_bbox(label_file, img_w, img_h)
        cls_gts.append(class_map[fname])

        # ----------- GT MASK ----------
        mask_path = os.path.join(MASKS_DIR, name_no_ext + ".png") # Assuming .png, adjust if needed
        gt_mask = (np.array(Image.open(mask_path).resize((224, 224))) > 127).astype(bool)

        # Get predictions from the TensorFlow models
        pred_cls_prob, pred_box = modelA.predict(img_batch, verbose=0)
        pred_cls = (pred_cls_prob > 0.5).astype(int).item()
        pred_seg = modelC.predict(img_batch, verbose=0)
        pred_mask = (pred_seg.squeeze() > 0.5).astype(bool)

        # ----------- Metrics ----------
        bbox_ious.append(iou(pred_box, gt_box))
        cls_preds.append(pred_cls)

        iou_s, dice_s = seg_metrics(pred_mask, gt_mask)
        seg_ious.append(iou_s)
        seg_dices.append(dice_s)

    # =============================================================
    #                 PRINT DATASET RESULTS
    # =============================================================
    print(f"\n📌 RESULTS FOR: {dataset_name}")
    print("--------------------------------------------------")
    print(f"Bounding Box IoU (mean):       {np.mean(bbox_ious):.4f}")
    print(f"Classification Accuracy:        {accuracy_score(cls_gts, cls_preds):.4f}")
    print(f"Classification F1 (macro):      {f1_score(cls_gts, cls_preds, average='macro'):.4f}")
    print(f"Segmentation IoU (mean):        {np.mean(seg_ious):.4f}")
    print(f"Segmentation Dice (mean):       {np.mean(seg_dices):.4f}")
    print("--------------------------------------------------\n")

# =============================================================
#                 RUN EVALUATION FOR BOTH DATASETS
# =============================================================
if __name__ == "__main__":
    # Get the root of the 'ColonNet' project directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Project root detected at: {project_root}")
    DATASET_1 = os.path.join(project_root, "Test Dataset for Auto-WCEBleedGen Challenge version 2", "Test Dataset 1")
    DATASET_2 = os.path.join(project_root, "Test Dataset for Auto-WCEBleedGen Challenge version 2", "Test Dataset 2")
    evaluate_dataset(DATASET_1, "Test Dataset 1")
    evaluate_dataset(DATASET_2, "Test Dataset 2")
