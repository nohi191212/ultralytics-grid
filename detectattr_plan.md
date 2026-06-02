# DetectAttr 魔改方案

## 目标

在 YOLO26s 检测头基础上，为每个检测框额外预测三个属性：性别（gender）、人种（race）、体型（body_type）。

## 数据格式

```
class x_center y_center width height gender race body_type
```

每行 8 列，末尾三个整数分别代表性别、人种、体型类别。

## 预训练权重

`yolo26s-face.pt` — 已训练好的 Detect 检测权重，用作 DetectAttr 训练的初始化。

---

## 修改文件清单及改动内容

### 1. `ultralytics/data/utils.py` — 标签验证

**位置：** `verify_image_label()` 函数，约第196-265行。

**改动：**

- 第231行 `assert lb.shape[1] == 5` 改为 `assert lb.shape[1] == 8`（当存在属性列时）
- 将 `lb[:, 5:]`（性别、人种、体型）提取为 `attrs`，与 `lb[:, :5]`（cls+bbox）分开
- 返回值增加 `attrs` 字段

### 2. `ultralytics/data/dataset.py` — 数据集

**位置1：** `YOLODataset.cache_labels()` 第130-141行  
**改动：** 缓存时额外存储 `"gender":`, `"race":`, `"body_type":` 列

**位置2：** `YOLODataset.update_labels_info()` 第249-279行  
**改动：** 将 gender/race/body_type 从 label dict 透传到输出（与 cls 同级）

**位置3：** `YOLODataset.collate_fn()` 第282-308行  
**改动：** 将 `gender`, `race`, `body_type` 加入 batch 拼接（`torch.cat`）

### 3. `ultralytics/data/augment.py` — Format 变换

**位置：** `Format.__call__()` 约第2022-2105行  
**改动：** 在输出 dict 中增加 `"gender"`, `"race"`, `"body_type"` tensors

### 4. `ultralytics/nn/modules/head.py` — 新建 DetectAttr 头

仿照 `OBB`（cv4）→ `Pose`（cv4）→ `Segment`（cv4+proto）模式，新增：

```python
class DetectAttr(Detect):
    def __init__(self, nc=80, ng=2, nr=7, nb=2, reg_max=1, end2end=True, ch=()):
        # ng: 性别类别数, nr: 人种类别数, nb: 体型类别数
        super().__init__(nc, reg_max, end2end, ch)
        self.ng, self.nr, self.nb = ng, nr, nb
        self.no = nc + reg_max * 4 + ng + nr + nb

        c4 = max(ch[0] // 4, max(ng, nr, nb))
        self.cv4 = nn.ModuleList(...)  # gender
        self.cv5 = nn.ModuleList(...)  # race
        self.cv6 = nn.ModuleList(...)  # body_type
        # + end2end 复制
```

**需重写的方法：**

- `one2many` / `one2one` properties
- `forward_head()` — 返回额外的 gender/race/body logits
- `_inference()` — 拼接所有输出
- `postprocess()` — 拆分并处理属性
- `bias_init()` — 初始化新分支 bias
- `fuse()` — 清理 one2many 分支

### 5. `ultralytics/utils/loss.py` — 新建 v8DetectionAttrLoss

```python
class v8DetectionAttrLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        head = model.model[-1]  # DetectAttr
        self.ng, self.nr, self.nb = head.ng, head.nr, head.nb
        self.no = head.no
        # 属性分类用 BCEWithLogitsLoss（与 cls 一致）
        self.bce_attr = nn.BCEWithLogitsLoss(reduction="none")
        self.ce_attr = nn.CrossEntropyLoss(reduction="none")
```

**核心逻辑：** `get_assigned_targets_and_loss()` 中：

- 从 `preds["gender"]`, `preds["race"]`, `preds["body_type"]` 获取 logits
- 从 `batch["gender"]`, `batch["race"]`, `batch["body_type"]` 获取 targets
- 通过 TAL assigner 将 targets 分配到 anchor，**只对正样本（fg_mask）计算属性 loss**
- loss 从 3 项变为 6 项：`[box, cls, dfl, gender, race, body_type]`

### 6. `ultralytics/cfg/models/26/yolo26-attr.yaml` — 模型配置

复制 `yolo26.yaml`，最后一行改为：

```yaml
- [[16, 19, 22], 1, DetectAttr, [nc, ng, nr, nb]]
```

根据数据统计设定 ng=2, nr=7, nb=2（可调）。

### 7. `ultralytics/nn/tasks.py` — 新建 AttrDetectionModel

```python
class AttrDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26-attr.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return E2ELoss(self, v8DetectionAttrLoss) if getattr(self, "end2end", False) else v8DetectionAttrLoss(self)
```

同时在 `parse_model()` 中确保 `DetectAttr` 被正确识别和构建。

### 8. 推理接口适配

**文件：** `ultralytics/engine/results.py` + predictor  
**改动：**

- `Results` 对象增加 `gender`, `race`, `body_type` 属性字段
- `postprocess()` 将输出 tensor 拆分并解析属性

### 9. `__init__.py` 导出链

- `ultralytics/nn/modules/__init__.py`：导出 `DetectAttr`
- `ultralytics/nn/tasks.py`：导入 `DetectAttr`
- 确保模型注册可用

---

## 实现顺序

| 顺序 | 文件                 | 说明                            |
| ---- | -------------------- | ------------------------------- |
| 1    | `data/utils.py`      | 标签解析（基础依赖）            |
| 2    | `data/dataset.py`    | Dataset 数据流                  |
| 3    | `data/augment.py`    | Format 变换                     |
| 4    | `nn/modules/head.py` | DetectAttr 头（核心）           |
| 5    | `utils/loss.py`      | 多属性 Loss                     |
| 6    | `nn/tasks.py` + YAML | 模型注册 + 配置                 |
| 7    | `__init__.py` 导出   | 注册导出                        |
| 8    | 推理/验证接口        | Results + predictor + validator |
