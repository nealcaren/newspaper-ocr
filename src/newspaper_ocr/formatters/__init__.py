from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.formatters.text import TextFormatter
from newspaper_ocr.formatters.json_fmt import JsonFormatter
from newspaper_ocr.formatters.hocr import HocrFormatter
from newspaper_ocr.registry import Registry

FORMATTERS = Registry("formatter")
FORMATTERS.register("text", TextFormatter)
FORMATTERS.register("json", JsonFormatter)
FORMATTERS.register("hocr", HocrFormatter)
