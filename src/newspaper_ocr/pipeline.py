from __future__ import annotations
from pathlib import Path
from PIL import Image
from newspaper_ocr.models import PageLayout
from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.recognizers.base import LineRecognizer, RegionRecognizer
from newspaper_ocr.formatters.base import Formatter


class Pipeline:
    def __init__(
        self,
        detector: Detector | None = None,
        recognizer: LineRecognizer | RegionRecognizer | None = None,
        formatter: Formatter | None = None,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.formatter = formatter

    def run(self, image: Image.Image) -> str:
        """Run the full pipeline on a PIL Image."""
        layout = self.detector.detect(image)

        if hasattr(self, 'layout_processor'):
            layout = self.layout_processor.process(layout)

        if isinstance(self.recognizer, LineRecognizer):
            for region in layout.regions:
                region.lines = self.recognizer.recognize_batch(region.lines)
                region.text = " ".join(line.text for line in region.lines if line.text)
        elif isinstance(self.recognizer, RegionRecognizer):
            for region in layout.regions:
                region = self.recognizer.recognize(region)

        return self.formatter.format(layout)

    def ocr(self, path: str | Path, output: str | None = None) -> str:
        """Run OCR on an image file."""
        image = Image.open(str(path)).convert("RGB")
        return self.run(image)

    def ocr_batch(self, paths: list[str | Path]) -> list[str]:
        """Run OCR on multiple image files."""
        return [self.ocr(p) for p in paths]
