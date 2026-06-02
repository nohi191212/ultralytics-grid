# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from ultralytics.data import build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.regress.train import HeightRegressionDataset
from ultralytics.utils import LOGGER


class RegressionValidator(BaseValidator):
    """Validator for normalized scalar height regression."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "regress"
        self.preds = []
        self.targets = []
        self.metrics = SimpleNamespace(keys=["metrics/mae_m", "metrics/rmse_m", "metrics/height_over_2m_acc"])

    def get_desc(self):
        return ("%22s" + "%11s" * 3) % ("height", "mae_m", "rmse_m", "over_2m_acc")

    def init_metrics(self, model):
        inner = getattr(model, "model", model)
        seq = getattr(inner, "model", inner)
        head = seq[-1] if hasattr(seq, "__getitem__") else None
        self.max_value = float(getattr(head, "max_value", 30.0))
        self.preds = []
        self.targets = []

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["height_m"] = batch["height_m"].to(self.device)
        return batch

    def postprocess(self, preds):
        y = preds[0] if isinstance(preds, (tuple, list)) else preds
        return y.sigmoid() * self.max_value if y.max() > 1.0 or y.min() < 0.0 else y * self.max_value

    def update_metrics(self, preds, batch):
        self.preds.append(preds.detach().view(-1).cpu())
        self.targets.append(batch["height_m"].detach().view(-1).cpu())

    def get_stats(self):
        if not self.preds:
            return {"metrics/mae_m": 0.0, "metrics/rmse_m": 0.0, "metrics/height_over_2m_acc": 0.0, "fitness": 0.0}
        pred = torch.cat(self.preds)
        target = torch.cat(self.targets)
        err = pred - target
        mae = err.abs().mean().item()
        rmse = math.sqrt(err.square().mean().item())
        over_2m_acc = ((pred >= 2.0) == (target >= 2.0)).float().mean().item()
        return {
            "metrics/mae_m": mae,
            "metrics/rmse_m": rmse,
            "metrics/height_over_2m_acc": over_2m_acc,
            "fitness": -mae,
        }

    def print_results(self):
        stats = self.get_stats()
        LOGGER.info(
            "Height regression — mae_m: {:.4f}  rmse_m: {:.4f}  over_2m_acc: {:.4f}".format(
                stats["metrics/mae_m"], stats["metrics/rmse_m"], stats["metrics/height_over_2m_acc"]
            )
        )

    def build_dataset(self, img_path, mode="val", batch=None):
        return HeightRegressionDataset(self.data, mode, imgsz=self.args.imgsz)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path, batch=batch_size)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
