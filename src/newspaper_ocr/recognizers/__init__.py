from newspaper_ocr.recognizers.base import LineRecognizer, RegionRecognizer
from newspaper_ocr.registry import Registry

RECOGNIZERS = Registry("recognizer")

try:
    from newspaper_ocr.recognizers.tesseract import TesseractRecognizer
    RECOGNIZERS.register("tesseract", TesseractRecognizer)
except ImportError:
    pass

try:
    from newspaper_ocr.recognizers.tesserocr_backend import TesserocrRecognizer
    RECOGNIZERS.register("tesserocr", TesserocrRecognizer)
except ImportError:
    pass

try:
    from newspaper_ocr.recognizers.effocr import EffocrRecognizer
    RECOGNIZERS.register("effocr", EffocrRecognizer)
except ImportError:
    pass
