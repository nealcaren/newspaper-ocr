from __future__ import annotations
from pathlib import Path
from PIL import Image
from newspaper_ocr import chunking
from newspaper_ocr.models import Region, PageLayout
from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.recognizers.base import LineRecognizer, RegionRecognizer
from newspaper_ocr.formatters.base import Formatter

#: Region statuses that make a region eligible for fallback re-OCR.
#:   no-loss  — nothing usable to preserve, so any usable fallback read wins
#:   partial  — real but degraded text, so only a clean fallback read replaces it
_FALLBACK_NO_LOSS = {"timeout", "error"}
_FALLBACK_PARTIAL = {"repetition", "chunked_partial"}


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
        chunk_tall_regions: bool = False,
        chunk_height: int = chunking.CHUNK_HEIGHT,
        chunk_overlap: int = chunking.CHUNK_OVERLAP,
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

        # Tall-region splitting for region-level recognizers (see _chunk_region).
        self.chunk_tall_regions = chunk_tall_regions
        self.chunk_height = chunk_height
        self.chunk_overlap = chunk_overlap

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

    def _chunk_region(self, region: Region) -> Region:
        """Re-OCR a tall region by splitting it into vertical bands.

        Called when the primary recognizer timed out on a region taller than
        ``chunk_height``. Each band is OCR'd with the same recognizer and the
        texts are stitched back together. Status becomes ``ok`` if every band
        was read, ``chunked_partial`` if some band timed out but others produced
        text, or ``timeout`` if nothing came back.
        """
        width, height = region.image.size
        spans = chunking.chunk_spans(height, self.chunk_height, self.chunk_overlap)

        texts: list[str] = []
        any_timeout = False
        for y0, y1 in spans:
            band = Region(
                bbox=region.bbox,
                image=region.image.crop((0, y0, width, y1)),
                label=region.label,
            )
            band = self.recognizer.recognize(band)
            if band.status == "timeout":
                any_timeout = True
            elif band.text:
                texts.append(band.text)

        if not texts:
            return region  # keep the primary's timeout text/status
        region.text = chunking.merge_chunk_texts(texts)
        region.status = "chunked_partial" if any_timeout else "ok"
        return region

    def _apply_region_fallback(self, region: Region) -> Region:
        """Re-OCR a failed region with the fallback recognizer, do no harm.

        For no-loss statuses (timeout/error) any usable fallback read is taken;
        for partial statuses (repetition/chunked_partial) the fallback read only
        replaces the primary text if it is a clean ``ok``. The original text is
        preserved in ``region.text_primary`` so the swap is reversible.
        """
        no_loss = region.status in _FALLBACK_NO_LOSS
        if not (no_loss or region.status in _FALLBACK_PARTIAL):
            return region

        result = self.fallback.recognize(
            Region(bbox=region.bbox, image=region.image, label=region.label)
        )
        if not result.text.strip() or result.status in ("timeout", "error"):
            return region  # fallback gave nothing usable

        accept = result.status in ("ok", "repetition") if no_loss else result.status == "ok"
        if accept:
            region.text_primary = region.text
            region.text = result.text
            region.status = result.status
            region.engine = type(self.fallback).__name__
        return region

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
                region = self.recognizer.recognize(region)
                # Escalation ladder: primary -> chunked re-OCR (same model, for
                # a tall region that timed out) -> fallback recognizer.
                if (
                    self.chunk_tall_regions
                    and region.status == "timeout"
                    and region.image is not None
                    and region.image.size[1] > self.chunk_height
                ):
                    region = self._chunk_region(region)
                if isinstance(self.fallback, RegionRecognizer):
                    region = self._apply_region_fallback(region)
                layout.regions[i] = region

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
