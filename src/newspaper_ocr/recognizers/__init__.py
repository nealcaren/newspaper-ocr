from newspaper_ocr.recognizers.base import LineRecognizer, RegionRecognizer
from newspaper_ocr.registry import Registry

RECOGNIZERS = Registry("recognizer")

try:
    from newspaper_ocr.recognizers.tesseract import TesseractRecognizer
    RECOGNIZERS.register("tesseract", TesseractRecognizer)
    RECOGNIZERS.register("tesseract-news", lambda **kw: TesseractRecognizer(model="news_combo_fast", **kw))
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

try:
    from newspaper_ocr.recognizers.glm_ocr import GlmOcrRecognizer
    RECOGNIZERS.register("glm-ocr", GlmOcrRecognizer)
except ImportError:
    pass
