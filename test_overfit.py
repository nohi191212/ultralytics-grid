# test_overfit.py — 快速过拟合测试，验证 loss 是否正常下降
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ultralytics"))

from ultralytics import YOLO

MODEL_CFG = "ultralytics/cfg/models/26/yolo26-attr.yaml"
PRETRAINED = None  # 跳过预训练，从零训练测试 loss 是否正常
DATA = r"C:\Users\csh10\Desktop\ths\OmniRobot\hybrid_test\data.yaml"
PROJECT = "runs/test_overfit"

model = YOLO(MODEL_CFG, task="detectattr")
if PRETRAINED:
    model.load(PRETRAINED)

results = model.train(
    data=DATA,
    epochs=100,
    imgsz=320,
    batch=8,
    device="cpu",
    workers=0,
    lr0=0.01,
    lrf=0.1,
    warmup_epochs=2,
    project=PROJECT,
    name="exp",
    exist_ok=True,
)

# 在训练集上推理
best = Path(PROJECT) / "exp" / "weights" / "best.pt"
model2 = YOLO(str(best))
metrics = model2.val(data=DATA, imgsz=320, device="cpu")
print("\n========== Overfitting test results ==========")
print("Validation metrics:", metrics)

# 在训练集上推理几张图看看效果
results_pred = model2.predict(
    source=r"C:\Users\csh10\Desktop\ths\OmniRobot\hybrid_test\images",
    imgsz=320,
    device="cpu",
    save=True,
    project=PROJECT,
    name="pred",
    exist_ok=True,
)
print(f"Predictions saved to {PROJECT}/pred")
