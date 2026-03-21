from newspaper_ocr.recognizers.base import LineRecognizer, RegionRecognizer
from newspaper_ocr.registry import Registry

RECOGNIZERS = Registry("recognizer")

try:
    from newspaper_ocr.recognizers.tesseract import TesseractRecognizer
    RECOGNIZERS.register("tesseract", TesseractRecognizer)
except ImportError:
    pass
