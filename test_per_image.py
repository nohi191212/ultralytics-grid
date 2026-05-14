"""Per-image evaluation: mAP + gender/race/body_type accuracy."""
from pathlib import Path
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# --- config ---
BEST = "/mnt/HithinkOmniSSD/user_workspace/caisihang/project/ultralytics/runs/two_stage_0512/stage2_full/weights/last.pt"
DATA_YAML = "/mnt/HithinkOmniSSD/user_workspace/caisihang/dataset/OmniRobotFaceDetect/data.yaml"
VAL_IMG = Path("/mnt/dataset/OmniRobotFaceDetect/images/val")
VAL_LBL = Path("/mnt/dataset/OmniRobotFaceDetect/labels/val")
IMG_SIZE = 640
CONF = 0.2
IOU_NMS = 0.65
MATCH_IOU = 0.5
NG, NR, NB = 2, 7, 4  # from data.yaml

# --- load model ---
model = YOLO(BEST)
inner = model.model.model[-1]
ng, nr, nb = inner.ng, inner.nr, inner.nb

print(f"Model head: ng={ng} nr={nr} nb={nb}")
print(f"Val images: {sum(1 for _ in VAL_IMG.glob('*.*'))}")

# --- per-image collect ---
all_gts = {}          # image_stem -> list of gt boxes (x1,y1,x2,y2,gender,race,body)
all_pds = {}          # image_stem -> list of [x1,y1,x2,y2,conf,gender,race,body]
attr_match_m = {"gender": 0, "race": 0, "body_type": 0, "total": 0}  # matched only

for img_path in tqdm(sorted(VAL_IMG.glob("*.*"))):
    stem = img_path.stem
    lbl_path = VAL_LBL / f"{stem}.txt"

    # parse GT
    gts = []
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            cls, xc, yc, w, h = map(float, parts[:5])
            g_gender, g_race, g_body = map(int, parts[5:8])
            img = cv2.imread(str(img_path))
            h_img, w_img = img.shape[:2]
            xc *= w_img; yc *= h_img; w *= w_img; h *= h_img
            x1 = xc - w / 2; y1 = yc - h / 2; x2 = xc + w / 2; y2 = yc + h / 2
            gts.append((x1, y1, x2, y2, g_gender, g_race, g_body))
    all_gts[stem] = gts

    # inference
    results = model.predict(str(img_path), imgsz=IMG_SIZE, conf=CONF, iou=IOU_NMS, verbose=False)
    r = results[0]

    pds = []
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()   # (N, 4)
        confs = r.boxes.conf.cpu().numpy()    # (N,)
        gender = r.gender.cpu().numpy() if hasattr(r, 'gender') else np.zeros(len(boxes), dtype=int)
        race = r.race.cpu().numpy() if hasattr(r, 'race') else np.zeros(len(boxes), dtype=int)
        body_type = r.body_type.cpu().numpy() if hasattr(r, 'body_type') else np.zeros(len(boxes), dtype=int)
        for j in range(len(boxes)):
            pds.append([*boxes[j], confs[j], int(gender[j]), int(race[j]), int(body_type[j])])
    all_pds[stem] = pds

    # IoU match & attribute accuracy
    if gts and pds:
        gt_boxes = np.array([g[:4] for g in gts])
        pd_boxes = np.array([p[:4] for p in pds])

        # compute IoU matrix (N_gt, N_pd)
        x1 = np.maximum(gt_boxes[:, None, 0], pd_boxes[None, :, 0])
        y1 = np.maximum(gt_boxes[:, None, 1], pd_boxes[None, :, 1])
        x2 = np.minimum(gt_boxes[:, None, 2], pd_boxes[None, :, 2])
        y2 = np.minimum(gt_boxes[:, None, 3], pd_boxes[None, :, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        area_pd = (pd_boxes[:, 2] - pd_boxes[:, 0]) * (pd_boxes[:, 3] - pd_boxes[:, 1])
        iou = inter / (area_gt[:, None] + area_pd[None, :] - inter + 1e-16)

        # greedy match (sorted by IoU descending, > MATCH_IOU)
        gt_ids, pd_ids = np.where(iou > MATCH_IOU)
        order = iou[gt_ids, pd_ids].argsort()[::-1]
        gt_ids, pd_ids = gt_ids[order], pd_ids[order]
        matched_gt, matched_pd = set(), set()
        for gt_i, pd_i in zip(gt_ids, pd_ids):
            if gt_i not in matched_gt and pd_i not in matched_pd:
                matched_gt.add(gt_i)
                matched_pd.add(pd_i)
                g = gts[gt_i]
                p = pds[pd_i]
                if p[5] == g[4]:
                    attr_match_m["gender"] += 1
                if p[6] == g[5]:
                    attr_match_m["race"] += 1
                if p[7] == g[6]:
                    attr_match_m["body_type"] += 1
                attr_match_m["total"] += 1

# --- compute mAP ---
# Collect all predictions with their matched GT status
pd_records = []  # (conf, matched)
n_total_gt = 0
for stem, gts in all_gts.items():
    n_total_gt += len(gts)
    pds = all_pds[stem]
    if not gts or not pds:
        for p in pds:
            pd_records.append((p[4], False))
        continue
    gt_boxes = np.array([g[:4] for g in gts])
    pd_boxes = np.array([p[:4] for p in pds])
    x1 = np.maximum(gt_boxes[:, None, 0], pd_boxes[None, :, 0])
    y1 = np.maximum(gt_boxes[:, None, 1], pd_boxes[None, :, 1])
    x2 = np.minimum(gt_boxes[:, None, 2], pd_boxes[None, :, 2])
    y2 = np.minimum(gt_boxes[:, None, 3], pd_boxes[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    area_pd = (pd_boxes[:, 2] - pd_boxes[:, 0]) * (pd_boxes[:, 3] - pd_boxes[:, 1])
    iou = inter / (area_gt[:, None] + area_pd[None, :] - inter + 1e-16)
    gt_ids, pd_ids = np.where(iou > MATCH_IOU)
    order = iou[gt_ids, pd_ids].argsort()[::-1]
    gt_ids, pd_ids = gt_ids[order], pd_ids[order]
    matched_gt, matched_pd = set(), set()
    for gt_i, pd_i in zip(gt_ids, pd_ids):
        if gt_i not in matched_gt and pd_i not in matched_pd:
            matched_gt.add(gt_i)
            matched_pd.add(pd_i)
    for j, p in enumerate(pds):
        pd_records.append((p[4], j in matched_pd))

# sort by confidence descending for PR curve
pd_records.sort(key=lambda x: x[0], reverse=True)
tp = np.array([r[1] for r in pd_records], dtype=np.float64)
fp = 1 - tp
tp_cumsum = np.cumsum(tp)
fp_cumsum = np.cumsum(fp)

precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
recall = tp_cumsum / max(n_total_gt, 1)

# Compute AP (area under PR curve via 11-point interpolation)
ap50 = 0.0
for t in np.linspace(0, 1, 11):
    p_at_r = np.max(precision[recall >= t]) if np.any(recall >= t) else 0.0
    ap50 += p_at_r / 11.0

# --- print results ---
print(f"\n{'='*55}")
print(f"  Detection mAP@0.5  : {ap50:.4f}")
print(f"  Total GT boxes     : {n_total_gt}")
print(f"  Total PD boxes     : {len(pd_records)}")
print(f"{'='*55}")

if attr_match_m["total"] > 0:
    print(f"  Attribute accuracy (matched boxes: {attr_match_m['total']}):")
    print(f"    gender_acc   : {attr_match_m['gender'] / attr_match_m['total']:.4f}")
    print(f"    race_acc     : {attr_match_m['race'] / attr_match_m['total']:.4f}")
    print(f"    body_acc     : {attr_match_m['body_type'] / attr_match_m['total']:.4f}")
else:
    print("  No matched boxes for attribute evaluation.")
print(f"{'='*55}")
