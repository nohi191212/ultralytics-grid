# 魔改版 Ultralytics 项目结构

本文档记录当前 `/mnt/HithinkOmniSSD/user_workspace/caisihang/project/guizhou/code/ultralytics` 目录下的代码结构，以及相对原版 Ultralytics 增加/改造的核心模块。本文档只描述现状，不包含新的代码改动。

## 1. 顶层结构

```text
code/ultralytics/
├── ultralytics/                 # Ultralytics Python 包主体
├── docs/                        # 原版文档
├── examples/                    # 原版示例
├── tests/                       # 原版测试
├── docker/                      # 原版 Docker 配置
├── train_attr.py                # DetectAttr 训练入口，偏旧版/独立脚本
├── train_attr_resume.py         # DetectAttr 断点恢复训练脚本
├── train_two_stage.py           # 两阶段训练/实验脚本
├── train.sh                     # 训练 shell 入口
├── train_2.sh                   # 训练 shell 入口
├── train_resume.sh              # 恢复训练 shell 入口
├── move_and_eval.py             # 权重移动和评估辅助脚本
├── test.py                      # 测试脚本
├── test_overfit.py              # 过拟合/小数据测试脚本
├── test_per_image.py            # 单图级测试脚本
├── detectattr_plan.md           # DetectAttr 改造规划记录
├── pyproject.toml               # Python 包配置
├── README.md                    # 原版说明
└── README.zh-CN.md              # 原版中文说明
```

## 2. 核心包结构

```text
ultralytics/
├── __init__.py
├── cfg/
│   ├── __init__.py              # 注册任务集合，包含 detectattr、regress
│   ├── default.yaml             # 默认训练/推理参数
│   └── models/
│       └── 26/
│           ├── yolo26.yaml      # YOLO26 检测模型
│           ├── yolo26s-grid.yaml
│           ├── yolo26s-attr.yaml
│           ├── yolo26s-power.yaml
│           ├── yolo26s-scene.yaml
│           ├── yolo26-cls.yaml
│           ├── yolo26-obb.yaml
│           ├── yolo26-p2.yaml
│           ├── yolo26-p6.yaml
│           ├── yolo26-pose.yaml
│           └── yolo26-seg.yaml
├── data/
│   ├── build.py                 # dataloader/dataset 构建
│   ├── dataset.py               # YOLODataset，支持 attr 标签字段
│   ├── augment.py               # 数据增强
│   ├── base.py
│   ├── loaders.py
│   └── utils.py                 # 数据集检查、yaml 读取等
├── engine/
│   ├── model.py                 # 通用 Model 基类
│   ├── trainer.py               # BaseTrainer，已兼容 regress/detectattr 数据检查
│   ├── validator.py             # BaseValidator，已兼容 regress yaml
│   ├── predictor.py
│   ├── results.py               # Results，承载属性输出字段
│   └── exporter.py
├── models/
│   └── yolo/
│       ├── model.py             # YOLO.task_map，注册 detectattr/regress
│       ├── detect/              # 原版检测任务
│       ├── classify/            # 原版分类任务
│       ├── segment/             # 原版分割任务
│       ├── pose/                # 原版姿态任务
│       ├── obb/                 # 原版旋转框任务
│       ├── detectattr/          # 新增：检测 + 属性任务
│       └── regress/             # 新增：图像级标量回归任务
├── nn/
│   ├── tasks.py                 # 模型类注册：AttrDetectionModel、RegressionModel
│   └── modules/
│       ├── head.py              # DetectAttr、Regress 等 head 实现
│       └── __init__.py          # 导出 DetectAttr、Regress
└── utils/
    ├── loss.py                  # v8DetectionAttrLoss、HeightRegressionLoss
    ├── metrics.py
    ├── nms.py
    ├── ops.py
    ├── plotting.py
    └── torch_utils.py
```

## 3. 自定义任务一：detectattr

`detectattr` 是当前贵州电网第三方检测里用于“目标检测 + 属性分类”的任务链。

```text
ultralytics/models/yolo/detectattr/
├── __init__.py
├── train.py                     # AttrDetectionTrainer
├── val.py                       # AttrDetectionValidator
└── predict.py                   # AttrDetectionPredictor
```

### 关键文件

```text
ultralytics/cfg/models/26/yolo26s-grid.yaml
ultralytics/cfg/models/26/yolo26s-attr.yaml
ultralytics/nn/modules/head.py
ultralytics/nn/tasks.py
ultralytics/utils/loss.py
ultralytics/models/yolo/model.py
ultralytics/models/yolo/detectattr/train.py
ultralytics/models/yolo/detectattr/val.py
ultralytics/models/yolo/detectattr/predict.py
```

### 主要职责

- `yolo26s-grid.yaml`
  - 贵州项目当前 attr 任务使用的模型结构。
  - `nc: 9`，对应电网场景中的 9 类目标。
  - `attr_names` 包含 7 个属性：
    - `person_work_state`
    - `helmet`
    - `safety_belt`
    - `crossing_fence`
    - `on_ladder`
    - `ladder_held`
    - `crane_under_person`
  - `attr_dims: [2, 2, 2, 2, 2, 2, 2]`，每个属性都是二分类。
  - head 最后一层为 `DetectAttr`。

- `ultralytics/nn/modules/head.py`
  - 新增/改造 `DetectAttr`。
  - 继承 `Detect`。
  - 在检测框和类别分支之外，额外维护 `cv_attrs` 属性分支。
  - 支持新版参数形式：`DetectAttr(nc, attr_dims, reg_max, end2end, ch)`。
  - 也兼容旧版人脸属性形式：`DetectAttr(nc, ng, nr, nb, reg_max, end2end, ch)`。
  - 推理时输出格式大致为：
    ```text
    boxes(4) + class_scores(nc) + attr_logits(sum(attr_dims))
    ```

- `ultralytics/nn/tasks.py`
  - 新增/改造 `AttrDetectionModel`。
  - 从 yaml 中读取 `attr_names`、`attr_dims`。
  - 初始化属性类别名。
  - `init_criterion()` 返回 `v8DetectionAttrLoss`。

- `ultralytics/utils/loss.py`
  - 新增/改造 `v8DetectionAttrLoss`。
  - 先走 YOLO 检测分配逻辑，得到前景 anchor 和匹配到的 GT。
  - 对前景 anchor 计算属性交叉熵。
  - 标签中属性值为 `-1` 时忽略该属性损失。
  - 支持通过超参为不同属性设置 loss 权重。

- `ultralytics/models/yolo/detectattr/train.py`
  - `AttrDetectionTrainer`。
  - 构建检测数据集。
  - 将 batch 中的 `attrs`、`gender`、`race`、`body_type` 移到训练设备。
  - 根据 data yaml 中的 `attrs` 字段设置模型属性名和属性维度。

- `ultralytics/models/yolo/detectattr/val.py`
  - `AttrDetectionValidator`。
  - 复用检测 mAP 统计。
  - 额外通过 IoU 匹配统计属性准确率。
  - 输出 `metrics/{attr_name}_acc` 和综合 `fitness`。

- `ultralytics/models/yolo/detectattr/predict.py`
  - `AttrDetectionPredictor`。
  - NMS 后解析属性 logits。
  - 将属性 argmax 结果写入 `Results(..., attrs=attrs)`。

## 4. 自定义任务二：regress

`regress` 是当前用于图像级标量回归的任务链，例如电力安全中的人员高度/距离等归一化标量任务。

```text
ultralytics/models/yolo/regress/
├── __init__.py
├── train.py                     # RegressionTrainer、HeightRegressionDataset
├── val.py                       # RegressionValidator
└── predict.py                   # RegressionPredictor
```

### 关键文件

```text
ultralytics/cfg/models/26/yolo26s-power.yaml
ultralytics/nn/modules/head.py
ultralytics/nn/tasks.py
ultralytics/utils/loss.py
ultralytics/models/yolo/regress/train.py
ultralytics/models/yolo/regress/val.py
ultralytics/models/yolo/regress/predict.py
```

### 主要职责

- `yolo26s-power.yaml`
  - 图像级回归模型配置。
  - head 使用 `Regress`。

- `ultralytics/nn/modules/head.py`
  - 新增/改造 `Regress`。
  - 对输入 feature 做卷积、全局池化和线性输出。
  - 训练返回 raw logits。
  - 推理返回 sigmoid 后的归一化标量。

- `ultralytics/nn/tasks.py`
  - 新增/改造 `RegressionModel`。
  - 使用 `HeightRegressionLoss`。

- `ultralytics/utils/loss.py`
  - 新增 `HeightRegressionLoss`。
  - 对 sigmoid 后的预测值和 `height_norm` 做 Smooth L1 损失。

- `ultralytics/models/yolo/regress/train.py`
  - 定义 `HeightRegressionDataset`。
  - 定义 `RegressionTrainer`。
  - 从 yaml/CSV 形式的数据配置构建训练集和验证集。

- `ultralytics/models/yolo/regress/val.py`
  - 定义 `RegressionValidator`。
  - 统计回归验证指标。

- `ultralytics/models/yolo/regress/predict.py`
  - 定义 `RegressionPredictor`。
  - 输出归一化标量预测。

## 5. 任务注册链路

```text
ultralytics/cfg/__init__.py
└── TASKS 增加 detectattr、regress

ultralytics/models/yolo/__init__.py
└── import detectattr、regress

ultralytics/models/yolo/model.py
└── YOLO.task_map 增加：
    ├── detectattr -> AttrDetectionModel / AttrDetectionTrainer / AttrDetectionValidator / AttrDetectionPredictor
    └── regress    -> RegressionModel     / RegressionTrainer     / RegressionValidator     / RegressionPredictor

ultralytics/nn/modules/__init__.py
└── 导出 DetectAttr、Regress

ultralytics/nn/tasks.py
└── parse_model 支持 DetectAttr、Regress
```

## 6. 贵州项目相关模型配置

```text
ultralytics/cfg/models/26/
├── yolo26.yaml                  # 普通检测模型
├── yolo26s-grid.yaml            # 电网安全检测 + 属性模型，当前 attr 任务主配置
├── yolo26s-attr.yaml            # 早期/通用 DetectAttr 配置，偏人脸属性形式
├── yolo26s-power.yaml           # 图像级标量回归配置
└── yolo26s-scene.yaml           # 场景分类配置
```

## 7. 数据格式适配

### detectattr 数据

贵州项目的 attr 数据集 yaml 形态大致如下：

```yaml
path: ...
train: images/train
val: images/val
test: images/test
names:
  0: pole
  1: electroscope
  2: fence
  3: warning_sign
  4: electric_box
  5: person
  6: crane_arm
  7: crane
  8: ladder
attrs:
  - name: person_work_state
    classes: [not_working, working_on_pole]
    applies_to: [person]
```

YOLO label 每一行在标准检测字段后追加属性字段：

```text
class_id x_center y_center width height attr_0 attr_1 ... attr_n
```

其中：

- bbox 仍是 YOLO normalized xywh。
- 属性值为 `0..num_classes-1`。
- 属性值为 `-1` 表示该目标不适用该属性，训练 loss 中会忽略。

### regress 数据

`regress` 任务使用 yaml 指向图像和标量标签，内部由 `HeightRegressionDataset` 读取，训练目标字段为 `height_norm`。

## 8. 外部训练脚本

这些脚本位于 `code/ultralytics/` 根目录，主要用于快速训练/恢复训练：

```text
train_attr.py
train_attr_resume.py
train_two_stage.py
train.sh
train_2.sh
train_resume.sh
move_and_eval.py
test.py
test_overfit.py
test_per_image.py
```

当前贵州项目更推荐使用上层训练入口：

```text
project/guizhou/code/grid_safety/training/start_training.py
project/guizhou/code/grid_safety/training/train_common.py
project/guizhou/code/grid_safety/training/train_grid_attr.py
```

上层入口会把本地 `code/ultralytics` 插入 `sys.path`，然后通过：

```python
YOLO(config, task="detectattr")
```

调用本 fork 内注册的 `detectattr` 任务。

## 9. 当前 attr head 现状说明

当前 `DetectAttr` 的属性分支位于：

```text
ultralytics/nn/modules/head.py
```

实现方式是：

- 每个检测尺度都有对应的属性卷积分支。
- 属性 logits 与检测候选位置一一对应。
- loss 里只对正样本 anchor 计算属性分类损失。
- 推理时 NMS 后保留对应候选位置的属性 logits。

注意：本文档生成时，尚未把属性分类头改成“bbox 上下左右各扩张 50% 的 ROI 特征”。如果后续要实现该需求，主要改动点预计会落在：

```text
ultralytics/nn/modules/head.py
ultralytics/utils/loss.py
ultralytics/models/yolo/detectattr/predict.py
```

具体需要保证训练和推理使用一致的 ROI 规则。
