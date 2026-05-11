# train_attr.py — YOLO26s DetectAttr 训练脚本
import sys
from pathlib import Path

# 确保 ultralytics 源码在搜索路径中
sys.path.insert(0, str(Path(__file__).parent / "ultralytics"))

from ultralytics import YOLO

# ========== 配置 ==========
MODEL_CFG = "ultralytics/ultralytics/cfg/models/26/yolo26-attr.yaml"  # 模型结构
PRETRAINED = "yolo26s-face.pt"  # 预训练权重
DATA = r"C:\Users\csh10\Desktop\ths\OmniRobot\hybrid_test\data.yaml"
PROJECT = "runs/attr_train"  # 输出目录

# ========== 训练 ==========
model = YOLO(MODEL_CFG, task="detectattr")
model.load(PRETRAINED)  # 加载预训练 backbone + detect head

results = model.train(
    data=DATA,
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU; CPU 设为 "cpu"
    workers=4,
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
