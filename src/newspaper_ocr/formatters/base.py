from __future__ import annotations
from abc import ABC, abstractmethod
from newspaper_ocr.models import PageLayout


class Formatter(ABC):
    @abstractmethod
    def format(self, layout: PageLayout) -> str:
        """Format OCR results into output string."""
