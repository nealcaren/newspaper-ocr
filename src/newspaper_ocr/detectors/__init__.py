from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.detectors.as_yolo import AsYoloDetector
from newspaper_ocr.registry import Registry

DETECTORS = Registry("detector")
DETECTORS.register("as_yolo", AsYoloDetector)

try:
    from newspaper_ocr.detectors.paddlex import PaddleXDetector
    DETECTORS.register("paddlex", PaddleXDetector)
except ImportError:
    pass
