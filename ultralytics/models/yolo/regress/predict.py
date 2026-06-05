# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor


class RegressionPredictor(BasePredictor):
    """Predictor for scalar YOLO regression models."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        y = preds[0] if isinstance(preds, (tuple, list)) else preds
        if y.max() > 1.0 or y.min() < 0.0:
            y = y.sigmoid()

        inner = getattr(self.model, "model", self.model)
        seq = getattr(inner, "model", inner)
        head = seq[-1] if hasattr(seq, "__getitem__") else None
        max_value = float(getattr(head, "max_value", 30.0))
        return y * max_value
