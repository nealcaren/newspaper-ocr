from __future__ import annotations
from pathlib import Path
from PIL import Image
from newspaper_ocr.models import Region, PageLayout
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
        text_cleaning: bool = True,
        spell_check: bool = False,
        device: str = "cpu",
        fallback: LineRecognizer | RegionRecognizer | str | None = None,
        fallback_threshold: float = 70,
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

        # Resolve fallback recognizer (optional)
        if isinstance(fallback, str):
            fb_cls = RECOGNIZERS.get(fallback)
            self.fallback = fb_cls()
        else:
            self.fallback = fallback

        # Threshold is on Tesseract's 0-100 scale; store as-is, compare against
        # line.confidence * 100 at runtime.
        self.fallback_threshold = fallback_threshold

        # Layout post-processing
        self.layout_processor = LayoutProcessor(enabled=layout_processing)

        # Text cleaning (dehyphenation + line joining)
        from newspaper_ocr.text_cleaner import TextCleaner
        self.text_cleaner = TextCleaner(enabled=text_cleaning)

        # Optional spell correction (off by default — it's aggressive)
        from newspaper_ocr.spell_checker import SpellChecker
        self.spell_checker = SpellChecker(enabled=spell_check)

    def _fallback_recognize_line(self, recognizer, line):
        """Use any recognizer (line or region) to re-recognize a single line."""
        if isinstance(recognizer, LineRecognizer):
            return recognizer.recognize(line)
        elif isinstance(recognizer, RegionRecognizer):
            # Wrap the line as a temporary single-line region so RegionRecognizers
            # (e.g. GlmOcrRecognizer) can handle it without modification.
            temp_region = Region(
                bbox=line.bbox,
                image=line.image,
                label="text",
                lines=[line],
            )
            result = recognizer.recognize(temp_region)
            line.text = result.text
            line.confidence = 1.0  # VLM fallback is trusted
            return line

    def run(self, image: Image.Image) -> str:
        layout = self.detector.detect(image)
        layout = self.layout_processor.process(layout)

        # Region-level recognition: recognizer has recognize_region and mode == "region"
        if (
            hasattr(self.recognizer, "recognize_region")
            and getattr(self.recognizer, "mode", "line") == "region"
        ):
            for region in layout.regions:
                self.recognizer.recognize_region(region)
        elif isinstance(self.recognizer, LineRecognizer):
            for region in layout.regions:
                region.lines = self.recognizer.recognize_batch(region.lines)
                region.text = " ".join(line.text for line in region.lines if line.text)
        elif isinstance(self.recognizer, RegionRecognizer):
            for i, region in enumerate(layout.regions):
                layout.regions[i] = self.recognizer.recognize(region)

        # Fallback: re-recognize low-confidence lines with the fallback recognizer.
        # Only applies when the primary recognizer is a LineRecognizer (so we have
        # per-line confidence scores) and a fallback has been configured.
        if self.fallback is not None and isinstance(self.recognizer, LineRecognizer):
            for region in layout.regions:
                for i, line in enumerate(region.lines):
                    if line.confidence * 100 < self.fallback_threshold:
                        region.lines[i] = self._fallback_recognize_line(
                            self.fallback, line
                        )
                # Rebuild region text after any fallback substitutions
                region.text = " ".join(
                    line.text for line in region.lines if line.text
                )

        # Text cleaning (dehyphenation, line joining) only for line-level recognizers.
        # Region-level recognizers (GLM-OCR, VLMs) already return clean text.
        if isinstance(self.recognizer, LineRecognizer):
            layout = self.text_cleaner.clean(layout)

        layout = self.spell_checker.check(layout)
        return self.formatter.format(layout)

    def ocr(self, path: str | Path, output: str | None = None) -> str:
        image = Image.open(str(path)).convert("RGB")
        return self.run(image)

    def ocr_batch(self, paths: list[str | Path]) -> list[str]:
        return [self.ocr(p) for p in paths]
