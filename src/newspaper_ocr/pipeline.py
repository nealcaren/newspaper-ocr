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
        skip_lines: bool = False,
        region_repair: bool = False,
    ):
        from newspaper_ocr.detectors import DETECTORS
        from newspaper_ocr.recognizers import RECOGNIZERS
        from newspaper_ocr.formatters import FORMATTERS
        from newspaper_ocr.layout_processor import LayoutProcessor

        # Resolve detector
        if isinstance(detector, str):
            det_cls = DETECTORS.get(detector)
            self.detector = det_cls(model_dir=model_cache_dir, skip_lines=skip_lines)
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

        # Optional post-recognition region repair (off by default — it spends
        # extra recognizer calls re-reading crops, and only dense multi-column
        # pages have the double-detected columns it exists to fix).
        from newspaper_ocr.region_repair import RegionRepair
        self.region_repair = RegionRepair(enabled=region_repair)

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

    def _recognize_crop(self):
        """A ``(crop) -> (text, status)`` callback, or None if OCR can't do crops.

        Region repair re-reads pieces of the page that were never detected as
        regions.  A line recognizer with no ``recognize_region`` cannot read one,
        and repair is explicitly built to skip its re-OCR passes rather than
        invent text, so it gets None instead of an adapter that would fail
        mid-page.
        """
        from newspaper_ocr.recognizers.base import recognize_crop

        recognizer = self.recognizer
        if not hasattr(recognizer, "recognize_region") and not isinstance(
            recognizer, RegionRecognizer
        ):
            return None
        return lambda crop: recognize_crop(recognizer, crop)

    def analyze(self, image: Image.Image) -> PageLayout:
        """Detect, recognize and post-process a page, returning the layout.

        This is :meth:`run` without the formatting step, for callers that need
        the regions themselves — a review site, an article-segmentation pass, or
        anything that wants to emit more than one representation of a page
        without OCRing it twice.
        """
        layout = self.detector.detect(image)
        layout = self.layout_processor.process(layout)

        # Stable per-page handles for downstream consumers, in reading order.
        for i, region in enumerate(layout.regions):
            if not region.id:
                region.id = f"r{i}"

        # Region-level recognition: recognizer has recognize_region and mode == "region"
        if (
            hasattr(self.recognizer, "recognize_region")
            and getattr(self.recognizer, "mode", "line") == "region"
        ):
            for region in layout.regions:
                self.recognizer.recognize_region(region)
        elif isinstance(self.recognizer, LineRecognizer):
            for region in layout.regions:
                if region.lines:
                    region.lines = self.recognizer.recognize_batch(region.lines)
                    region.text = " ".join(line.text for line in region.lines if line.text)
                elif hasattr(self.recognizer, "recognize_region") and region.image is not None:
                    # Regions without detected lines (ads, tables, etc.) —
                    # fall back to region-level OCR
                    self.recognizer.recognize_region(region)
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

        # Post-recognition repair: text-aware dedup, container splitting and
        # fragment merging.  It runs here, before text cleaning, because the two
        # re-OCR passes need the recognizer and the page image, and because
        # cleaning should see the regions a caller will actually get.
        layout = self.region_repair.repair(layout, self._recognize_crop())

        # Text cleaning (dehyphenation, line joining) only for line-level recognizers.
        # Region-level recognizers (GLM-OCR, VLMs) already return clean text.
        if isinstance(self.recognizer, LineRecognizer):
            layout = self.text_cleaner.clean(layout)

        layout = self.spell_checker.check(layout)
        return layout

    def run(self, image: Image.Image) -> str:
        """Analyze a page and render it with the configured formatter."""
        return self.formatter.format(self.analyze(image))

    def ocr(self, path: str | Path, output: str | None = None) -> str:
        image = Image.open(str(path)).convert("RGB")
        return self.run(image)

    def ocr_batch(self, paths: list[str | Path]) -> list[str]:
        return [self.ocr(p) for p in paths]

    def ocr_pdf(
        self,
        path: str | Path,
        rotate: int = 0,
        dpi: int = 300,
        pages: range | list[int] | None = None,
    ) -> list[str]:
        """OCR a multi-page PDF, returning one formatted result per page.

        Each page is taken from its largest embedded image when it has one (the
        first embedded image is often a scanning-service banner, not the page),
        and rendered at *dpi* otherwise.  ``rotate`` turns each page clockwise by
        0, 90, 180 or 270 degrees before layout detection — sideways broadsheets
        produce nothing usable otherwise.

        Use :func:`newspaper_ocr.pdf.page_images` directly to stream pages
        without holding every page's output in memory.
        """
        from newspaper_ocr.pdf import page_images

        return [
            self.run(image)
            for image in page_images(path, dpi=dpi, rotate=rotate, pages=pages)
        ]
