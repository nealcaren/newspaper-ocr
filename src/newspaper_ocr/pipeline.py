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
        detector: Detector | str = "as_yolo",
        recognizer: LineRecognizer | RegionRecognizer | str = "tesseract",
        output: Formatter | str = "text",
        recognizer_model: str | Path | None = None,
        model_cache_dir: str | Path | None = None,
        layout_processing: bool = True,
        device: str = "cpu",
    ):
        from newspaper_ocr.detectors import DETECTORS
        from newspaper_ocr.recognizers import RECOGNIZERS
        from newspaper_ocr.formatters import FORMATTERS
        from newspaper_ocr.layout_processor import LayoutProcessor

        # Resolve detector
        if isinstance(detector, str):
            det_cls = DETECTORS.get(detector)
            self.detector = det_cls(model_dir=model_cache_dir)
        else:
            self.detector = detector

        # Resolve recognizer
        if isinstance(recognizer, str):
            rec_cls = RECOGNIZERS.get(recognizer)
            rec_kwargs = {}
            if recognizer_model:
                # TesseractRecognizer uses "model", EffocrRecognizer uses "model_dir"
                import inspect
                params = inspect.signature(rec_cls.__init__).parameters
                if "model" in params:
                    rec_kwargs["model"] = str(recognizer_model)
                elif "model_dir" in params:
                    rec_kwargs["model_dir"] = str(recognizer_model)
            self.recognizer = rec_cls(**rec_kwargs)
        else:
            self.recognizer = recognizer

        # Resolve formatter
        if isinstance(output, str):
            fmt_cls = FORMATTERS.get(output)
            self.formatter = fmt_cls()
        else:
            self.formatter = output

        # Layout post-processing
        self.layout_processor = LayoutProcessor(enabled=layout_processing)

    def run(self, image: Image.Image) -> str:
        layout = self.detector.detect(image)
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
        image = Image.open(str(path)).convert("RGB")
        return self.run(image)

    def ocr_batch(self, paths: list[str | Path]) -> list[str]:
        return [self.ocr(p) for p in paths]
