# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import csv
from copy import copy
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from ultralytics.data import build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import RegressionModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class HeightRegressionDataset(Dataset):
    """CSV-backed dataset for cropped person height regression."""

    def __init__(self, data: dict, split: str, imgsz: int = 224):
        self.root = Path(data.get("path", "."))
        self.split = split
        self.imgsz = imgsz
        labels_pattern = data.get("labels", "labels/{split}.csv")
        self.csv_path = self.root / labels_pattern.format(split=split)
        self.samples = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.samples.append((self.root / row["image"], float(row["height_norm"]), float(row["height_m"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        path, height_norm, height_m = self.samples[i]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(cv2.resize(img, (self.imgsz, self.imgsz)), cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        return {
            "img": img,
            "cls": torch.zeros(1, dtype=torch.long),
            "height_norm": torch.tensor([height_norm], dtype=torch.float32),
            "height_m": torch.tensor([height_m], dtype=torch.float32),
            "im_file": str(path),
        }


class RegressionTrainer(BaseTrainer):
    """Trainer for scalar YOLO regression models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = overrides or {}
        overrides["task"] = "regress"
        overrides.setdefault("imgsz", 224)
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = ["loss"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = RegressionModel(cfg, nc=1, ch=self.data.get("channels", 3), verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        for p in model.parameters():
            p.requires_grad = True
        return model

    def set_model_attributes(self):
        self.model.names = {0: "height"}
        self.model.args = self.args

    def build_dataset(self, img_path, mode="train", batch=None):
        return HeightRegressionDataset(self.data, mode, imgsz=self.args.imgsz)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=mode == "train", rank=rank)

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["cls"] = batch["cls"].to(self.device)
        batch["height_norm"] = batch["height_norm"].to(self.device)
        batch["height_m"] = batch["height_m"].to(self.device)
        return batch

    def get_validator(self):
        self.loss_names = ["loss"]
        return yolo.regress.RegressionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        return dict(zip(keys, [round(float(loss_items), 5)]))
