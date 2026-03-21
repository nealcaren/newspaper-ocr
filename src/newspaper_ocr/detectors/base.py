from __future__ import annotations
from abc import ABC, abstractmethod
from PIL import Image
from newspaper_ocr.models import PageLayout


class Detector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> PageLayout:
        """Detect layout regions and lines in an image."""
