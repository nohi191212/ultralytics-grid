PATH = "/mnt/HithinkOmniSSD/user_workspace/caisihang/checkpoints/260515_detect_face/0514/exp/weights/last.pt"


#!/usr/bin/env python3
"""
功能：
1. 将指定目录的某个文件移动到 /mnt/workspace/checkpoint/ 下
2. 将评估结果写入 /mnt/workspace/inference/statistic.jsonl
"""

import argparse
import json
import os
import shutil
import sys


def move_file_to_checkpoint(source_path, checkpoint_dir="/mnt/workspace/checkpoint"):
    """将指定文件移动到checkpoint目录.

    Args:
        source_path: 源文件路径
        checkpoint_dir: 目标checkpoint目录

    Returns:
        移动后的文件完整路径
    """
    # 检查源文件是否存在
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"源文件不存在: {source_path}")

    # 确保目标目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 获取文件名
    filename = os.path.basename(source_path)

    # 目标路径
    dest_path = os.path.join(checkpoint_dir, filename)

    # 如果目标文件已存在，可以选择覆盖或重命名
    if os.path.exists(dest_path):
        # 方案1: 直接覆盖（取消注释下一行）
        # os.remove(dest_path)

        # 方案2: 添加时间戳避免冲突
        import time

        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}{ext}"
        dest_path = os.path.join(checkpoint_dir, filename)
        print(f"目标文件已存在，重命名为: {filename}")

    # 移动文件
    shutil.copy(source_path, dest_path)
    print(f"文件已成功移动: {source_path} -> {dest_path}")

    return dest_path


def write_evaluation_result(result_dir="/mnt/workspace/inference", result_file="statistic.jsonl", data_dict=None):
    """写入评估结果到JSONL文件.

    Args:
        result_dir: 结果存放目录
        result_file: 结果文件名
        data_dict: 要写入的字典数据
    """
    # 默认数据
    if data_dict is None:
        data_dict = {"Omni问财打点测试集v1_1-0-0": {"ap50": 0.6568}}

    # 确保目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 文件路径
    filepath = os.path.join(result_dir, result_file)

    # 写入JSONL格式（每行一个JSON对象）
    with open(filepath, "w", encoding="utf-8") as f:
        json_str = json.dumps(data_dict, ensure_ascii=False)
        f.write(json_str + "\n")

    print(f"评估结果已写入: {filepath}")
    print(f"内容: {json_str}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="移动模型文件并记录评估结果")
    parser.add_argument("--source", "-s", default=PATH, help="需要移动的源文件路径")
    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        default="/mnt/workspace/checkpoint",
        help="checkpoint目标目录 (默认: /mnt/workspace/checkpoint)",
    )
    parser.add_argument(
        "--result-dir", "-r", default="/mnt/workspace/inference", help="结果存放目录 (默认: /mnt/workspace/inference)"
    )
    parser.add_argument("--result-file", "-f", default="statistic.jsonl", help="结果文件名 (默认: statistic.jsonl)")
    parser.add_argument("--ap50", "-a", type=float, default=0.6568, help="AP50分数 (默认: 0.6568)")
    parser.add_argument("--dataset-name", "-d", default="Omni问财打点测试集v1_1-0-0", help="数据集名称")

    args = parser.parse_args()

    try:
        # 步骤1: 移动文件到checkpoint目录
        print("=" * 60)
        print("步骤1: 移动文件到checkpoint目录")
        print("=" * 60)
        move_file_to_checkpoint(args.source, args.checkpoint_dir)

        # 步骤2: 写入评估结果
        print("\n" + "=" * 60)
        print("步骤2: 写入评估结果")
        print("=" * 60)
        eval_data = {args.dataset_name: {"ap50": args.ap50}}
        write_evaluation_result(args.result_dir, args.result_file, eval_data)

        print("\n" + "=" * 60)
        print("所有操作完成!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
