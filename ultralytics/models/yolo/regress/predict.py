# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor


class RegressionPredictor(BasePredictor):
    """Predictor for scalar YOLO regression models."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        y = preds[0] if isinstance(preds, (tuple, list)) else preds
        return y.sigmoid()
