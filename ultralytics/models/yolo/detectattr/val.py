# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, nms, ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class AttrDetectionValidator(DetectionValidator):
    """A validator for YOLO detection models with gender, race, and body_type attribute prediction."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detectattr"

    def postprocess(self, preds):
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detectattr" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=False,
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path, batch=batch_size)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
