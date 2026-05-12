# train_attr.py — YOLO26s DetectAttr 训练脚本
import sys
from pathlib import Path
import argparse
import os
os.chdir(os.path.dirname(__file__))

from ultralytics import YOLO

argparser = argparse.ArgumentParser()

argparser.add_argument('--config', required=True)
argparser.add_argument('--pretrained', default=None)
argparser.add_argument('--data', required=True)
argparser.add_argument('--project', required=True)
argparser.add_argument('--device', required=True)

args = argparser.parse_args()

def parse_device(device_str):
    """解析设备并返回设备配置和 GPU 数量"""
    if device_str.lower() == "cpu":
        return "cpu", 0  # CPU 模式，GPU 数量为 0
    elif "," in device_str:
        gpu_ids = [int(x.strip()) for x in device_str.split(",")]
        return gpu_ids, len(gpu_ids)
    else:
        return int(device_str), 1

DEVICE, gpu_count = parse_device(args.device)
import multiprocessing
cpu_count = multiprocessing.cpu_count()
workers = max(4, gpu_count * 4) if gpu_count > 0 else 4
workers = min(workers, 4)  # 不超过 CPU 核心数

# ========== 配置 ==========
MODEL_CFG = args.config  # 模型结构
PRETRAINED = args.pretrained  # 预训练权重
DATA = args.data
PROJECT = args.project  # 输出目录

# ========== 训练 ==========
model = YOLO(MODEL_CFG, task="detectattr")
model.load(PRETRAINED)  # 加载预训练 backbone + detect head

results = model.train(
    data=DATA,
    epochs=100,
    imgsz=640,
    batch=32,
    device=DEVICE,  # GPU; CPU 设为 "cpu"
    workers=workers,
    lr0=0.002,
    lrf=0.01,
    warmup_epochs=3,
    close_mosaic=10,
    project=PROJECT,
    name="exp",
    exist_ok=True,
)

# ========== 评估 ==========
best = Path(PROJECT) / "exp" / "weights" / "best.pt"
model2 = YOLO(str(best))
metrics = model2.val(data=DATA, imgsz=640)
print("Validation metrics:", metrics)
