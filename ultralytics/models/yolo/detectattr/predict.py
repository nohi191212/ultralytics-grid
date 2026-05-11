# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class AttrDetectionPredictor(BasePredictor):
    """A predictor for YOLO detection models with gender, race, and body_type attribute prediction."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        save_feats = getattr(self, "_feats", None) is not None
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detectattr" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=False,
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        s = min(x.shape[1] for x in feat_maps)
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1)
             for x in feat_maps], dim=1
        )
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]

    def _get_inner_model(self):
        """Get the underlying task model, penetrating AutoBackend/PyTorchBackend wrappers."""
        m = self.model
        if hasattr(m, 'backend') and hasattr(m.backend, 'model'):
            return m.backend.model
        return m

    def construct_results(self, preds, img, orig_imgs):
        results = []
        inner = self._get_inner_model()
        head = inner.model[-1]
        ng, nr, nb = head.ng, head.nr, head.nb
        gender_names = getattr(inner, "gender_names", None)
        race_names = getattr(inner, "race_names", None)
        body_names = getattr(inner, "body_names", None)
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            if len(pred):
                extra = pred[:, 6:]
                gender_logits = extra[:, :ng]
                race_logits = extra[:, ng:ng + nr]
                body_logits = extra[:, ng + nr:ng + nr + nb]
                gender = gender_logits.argmax(dim=-1)
                race = race_logits.argmax(dim=-1)
                body_type = body_logits.argmax(dim=-1)
                results.append(
                    Results(
                        orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        gender=gender,
                        race=race,
                        body_type=body_type,
                        gender_names=gender_names,
                        race_names=race_names,
                        body_names=body_names,
                    )
                )
            else:
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names,
                                       gender_names=gender_names,
                                       race_names=race_names,
                                       body_names=body_names))
        return results
