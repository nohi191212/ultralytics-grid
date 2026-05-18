# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK, nms
from ultralytics.utils.metrics import box_iou


class AttrDetectionValidator(DetectionValidator):
    """A validator for YOLO detection models with gender, race, and body_type attribute prediction."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detectattr"
        self.attr_correct = {"gender": 0, "race": 0, "body_type": 0}
        self.attr_total = 0

    def init_metrics(self, model):
        super().init_metrics(model)
        self.attr_correct = {"gender": 0, "race": 0, "body_type": 0}
        self.attr_total = 0
        # Cache attribute class counts from data config
        self.ng = self.data.get("ng", 2)
        self.nr = self.data.get("nr", 7)
        self.nb = self.data.get("nb", 4)

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
        # DetectAttr.postprocess output: [boxes(4), scores(1), conf(1), gender(ng), race(nr), body(nb)]
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update detection metrics and track attribute accuracy."""
        has_attrs = "gender" in batch and "race" in batch and "body_type" in batch

        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)

            npr = pred["cls"].shape[0]
            ngt = pbatch["cls"].shape[0]

            # Compute detection tp matrix
            if npr == 0 or ngt == 0:
                self.metrics.update_stats(
                    {
                        "tp": np.zeros((npr, self.niou), dtype=bool),
                        "target_cls": pbatch["cls"].cpu().numpy(),
                        "target_img": np.unique(pbatch["cls"].cpu().numpy()),
                        "conf": np.zeros(0) if npr == 0 else pred["conf"].cpu().numpy(),
                        "pred_cls": np.zeros(0) if npr == 0 else pred["cls"].cpu().numpy(),
                        "im_name": Path(pbatch["im_file"]).name,
                    }
                )
                if self.args.plots:
                    self.confusion_matrix.process_batch(pred, pbatch, conf=self.args.conf)
                continue

            iou = box_iou(pbatch["bboxes"], pred["bboxes"])
            tp = self.match_predictions(pred["cls"], pbatch["cls"], iou).cpu().numpy()

            self.metrics.update_stats(
                {
                    "tp": tp,
                    "target_cls": pbatch["cls"].cpu().numpy(),
                    "target_img": np.unique(pbatch["cls"].cpu().numpy()),
                    "conf": pred["conf"].cpu().numpy(),
                    "pred_cls": pred["cls"].cpu().numpy(),
                    "im_name": Path(pbatch["im_file"]).name,
                }
            )

            if self.args.plots:
                self.confusion_matrix.process_batch(pred, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(
                        batch["img"][si], pbatch["im_file"], self.save_dir
                    )

            if self.args.save_json or self.args.save_txt:
                predn = self._prepare_pred(pred)
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

            # Track attribute accuracy via IoU matching
            if has_attrs and pred["extra"].shape[1] > 0:
                correct_class = pbatch["cls"].unsqueeze(1) == pred["cls"]
                iou_cls = iou * correct_class
                matched_pd = self._greedy_match(iou_cls)  # list length ngt, value = pd_idx or -1

                # Attribute GT indices in the flat batch tensors
                gt_mask = batch["batch_idx"] == si
                gt_gender = batch["gender"].view(-1)[gt_mask].long()  # (ngt,)
                gt_race = batch["race"].view(-1)[gt_mask].long()
                gt_body = batch["body_type"].view(-1)[gt_mask].long()

                extra = pred["extra"]  # (npr, total_attr)

                for gt_i, pd_i in enumerate(matched_pd):
                    if pd_i >= 0:
                        gender_logits = extra[pd_i, :self.ng]
                        race_logits = extra[pd_i, self.ng:self.ng + self.nr]
                        body_logits = extra[pd_i, self.ng + self.nr:]

                        if gender_logits.argmax() == gt_gender[gt_i]:
                            self.attr_correct["gender"] += 1
                        if race_logits.argmax() == gt_race[gt_i]:
                            self.attr_correct["race"] += 1
                        if body_logits.argmax() == gt_body[gt_i]:
                            self.attr_correct["body_type"] += 1
                        self.attr_total += 1

    @staticmethod
    def _greedy_match(iou: torch.Tensor) -> list[int]:
        """Greedy one-to-one matching: for each GT, pick the best IoU prediction above threshold 0.5.

        Args:
            iou: (N_gt, N_pd) IoU matrix pre-filtered by correct class.

        Returns:
            list[int]: length N_gt, each element is the matched prediction index or -1.
        """
        ngt = iou.shape[0]
        if iou.numel() == 0:
            return [-1] * ngt
        iou_np = iou.cpu().numpy()
        gt_ids, pd_ids = np.where(iou_np > 0.5)
        if len(gt_ids) == 0:
            return [-1] * ngt
        # Sort by IoU descending for greedy matching
        order = iou_np[gt_ids, pd_ids].argsort()[::-1]
        gt_ids, pd_ids = gt_ids[order], pd_ids[order]
        matched_gt, matched_pd = set(), set()
        result = [-1] * ngt
        for gt, pd in zip(gt_ids, pd_ids):
            if gt not in matched_gt and pd not in matched_pd:
                matched_gt.add(gt)
                matched_pd.add(pd)
                result[gt] = int(pd)
        return result

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics including attribute accuracy and custom fitness."""
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        results = self.metrics.results_dict

        # Attribute accuracy
        if self.attr_total > 0:
            results["metrics/gender_acc"] = self.attr_correct["gender"] / self.attr_total
            results["metrics/race_acc"] = self.attr_correct["race"] / self.attr_total
            results["metrics/body_acc"] = self.attr_correct["body_type"] / self.attr_total
        else:
            results["metrics/gender_acc"] = 0.0
            results["metrics/race_acc"] = 0.0
            results["metrics/body_acc"] = 0.0

        # Custom fitness: detection mAP (60%) + average attribute accuracy (40%)
        attr_acc_avg = (
            results["metrics/gender_acc"] + results["metrics/race_acc"] + results["metrics/body_acc"]
        ) / 3.0
        if self.attr_total > 0:
            results["fitness"] = (
                0.5 * results["metrics/mAP50-95(B)"]
                + 0.1 * results["metrics/mAP50(B)"]
                + 0.4 * attr_acc_avg
            )
        return results

    def gather_stats(self) -> None:
        """Gather detection and attribute statistics from all validation ranks."""
        super().gather_stats()
        if RANK == -1 or not dist.is_available() or not dist.is_initialized():
            return

        attr_stats = torch.tensor(
            [
                self.attr_correct["gender"],
                self.attr_correct["race"],
                self.attr_correct["body_type"],
                self.attr_total,
            ],
            device=self.device,
            dtype=torch.float64,
        )
        dist.reduce(attr_stats, dst=0, op=dist.ReduceOp.SUM)
        if RANK == 0:
            self.attr_correct["gender"] = int(attr_stats[0].item())
            self.attr_correct["race"] = int(attr_stats[1].item())
            self.attr_correct["body_type"] = int(attr_stats[2].item())
            self.attr_total = int(attr_stats[3].item())

    def print_results(self) -> None:
        """Print detection and attribute accuracy metrics."""
        super().print_results()
        if self.attr_total > 0:
            LOGGER.info(
                "Attribute accuracy — gender: %.4f  race: %.4f  body_type: %.4f  (matched samples: %d)"
                % (
                    self.attr_correct["gender"] / self.attr_total,
                    self.attr_correct["race"] / self.attr_total,
                    self.attr_correct["body_type"] / self.attr_total,
                    self.attr_total,
                )
            )

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path, batch=batch_size)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
