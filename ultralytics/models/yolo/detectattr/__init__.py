# Ultralytics YOLO detectattr module
from .predict import AttrDetectionPredictor
from .train import AttrDetectionTrainer
from .val import AttrDetectionValidator

__all__ = "AttrDetectionPredictor", "AttrDetectionTrainer", "AttrDetectionValidator"
