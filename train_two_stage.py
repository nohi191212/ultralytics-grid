# train_two_stage.py — YOLO26s DetectAttr 两阶段训练脚本
# Stage 1: 冻结 backbone，训练 neck + heads (warmup 属性头)
# Stage 2: 全解冻，继续训练至收敛
import sys
import os
import argparse
from pathlib import Path

os.chdir(os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).parent / "ultralytics"))

from ultralytics import YOLO

argparser = argparse.ArgumentParser(description="Two-stage DetectAttr training")
argparser.add_argument("--config", required=True, help="模型 YAML 路径")
argparser.add_argument("--pretrained", required=True, help="预训练权重路径")
argparser.add_argument("--data", required=True, help="数据集 data.yaml")
argparser.add_argument("--project", required=True, help="输出目录")
argparser.add_argument("--device", required=True, help="设备: cpu / 0 / 0,1")
argparser.add_argument("--stage1-epochs", type=int, default=5, help="Stage1 冻结 backbone 训练轮数")
argparser.add_argument("--stage2-epochs", type=int, default=100, help="Stage2 全解冻训练轮数")
argparser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
argparser.add_argument("--batch", type=int, default=16, help="Batch size")
args = argparser.parse_args()


def parse_device(device_str):
    if device_str.lower() == "cpu":
        return "cpu", 0
    elif "," in device_str:
        gpu_ids = [int(x.strip()) for x in device_str.split(",")]
        return gpu_ids, len(gpu_ids)
    else:
        return int(device_str), 1


DEVICE, gpu_count = parse_device(args.device)
workers = min(max(4, gpu_count * 4) if gpu_count > 0 else 4, 8)

# ==================== Stage 1: 冻结 backbone ====================
print("=" * 60)
print(f"Stage 1: Freeze backbone, train {args.stage1_epochs} epochs")
print(f"  config={args.config}, pretrained={args.pretrained}")
print("=" * 60)

model = YOLO(args.config, task="detectattr")
model.load(args.pretrained)

results_s1 = model.train(
    data=args.data,
    epochs=args.stage1_epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=DEVICE,
    workers=workers,
    lr0=0.01,          # 较高初始 LR，让属性头快速学习
    lrf=0.1,           # 最终 LR = 0.001
    warmup_epochs=1,
    freeze=11,         # 冻结 backbone (层 0-10)，neck + heads 可训练
    project=args.project,
    name="stage1_frozen",
    exist_ok=True,
)

# ==================== Stage 2: 全解冻 ====================
print("\n" + "=" * 60)
print(f"Stage 2: Unfreeze all, train {args.stage2_epochs} epochs")
print("=" * 60)

best_s1 = Path(args.project) / "stage1_frozen" / "weights" / "best.pt"
model2 = YOLO(str(best_s1))

results_s2 = model2.train(
    data=args.data,
    epochs=args.stage2_epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=DEVICE,
    workers=workers,
    lr0=0.005,         # 较低初始 LR，微调全模型
    lrf=0.01,          # 最终 LR = 5e-5
    warmup_epochs=2,
    freeze=0,           # 全解冻
    project=args.project,
    name="stage2_full",
    exist_ok=True,
)

# ==================== 评估 ====================
print("\n" + "=" * 60)
print("Final evaluation on best checkpoint")
print("=" * 60)

best_final = Path(args.project) / "stage2_full" / "weights" / "best.pt"
model3 = YOLO(str(best_final))
metrics = model3.val(data=args.data, imgsz=args.imgsz, device=DEVICE)
print("\nValidation metrics:", metrics)
