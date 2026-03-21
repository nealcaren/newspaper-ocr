from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.models import PageLayout


class TextFormatter(Formatter):
    def format(self, layout: PageLayout) -> str:
        return layout.text
