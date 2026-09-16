from __future__ import annotations
from pathlib import Path
from typing import Callable
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
        detector: Detector | str = "auto",
        recognizer: LineRecognizer | RegionRecognizer | str | Callable = "tesseract",
        output: Formatter | str = "text",
        recognizer_model: str | Path | None = None,
        model_cache_dir: str | Path | None = None,
        layout_processing: bool = True,
        text_cleaning: bool = True,
        spell_check: bool = False,
        device: str = "cpu",
        fallback: LineRecognizer | RegionRecognizer | str | Callable | None = None,
        fallback_threshold: float = 70,
        skip_lines: bool = False,
        chunk_tall_regions: bool = False,
        chunk_height: int = chunking.CHUNK_HEIGHT,
        chunk_overlap: int = chunking.CHUNK_OVERLAP,
        residual_ocr: bool | str = "auto",
    ):
        from newspaper_ocr.detectors import DETECTORS
        from newspaper_ocr.recognizers import RECOGNIZERS
        from newspaper_ocr.formatters import FORMATTERS
        from newspaper_ocr.layout_processor import LayoutProcessor

        # Resolve detector
        if isinstance(detector, str):
            name = self._resolve_detector_name(detector)
            det_cls = DETECTORS.get(name)
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
            self.recognizer = self._as_recognizer(recognizer)

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
        elif fallback is None:
            self.fallback = None
        else:
            self.fallback = self._as_recognizer(fallback)

        # A region-level primary can only fall back to a region-level recognizer:
        # the region path re-OCRs whole regions, and the line-level fallback path
        # never runs for a region primary — so a line fallback would be silently
        # ignored. Reject it loudly instead.
        if (
            self.fallback is not None
            and isinstance(self.recognizer, RegionRecognizer)
            and not isinstance(self.fallback, RegionRecognizer)
        ):
            raise ValueError(
                "A region-level recognizer needs a region-level fallback; "
                f"got fallback={type(self.fallback).__name__}. Use a "
                "RegionRecognizer such as 'paddleocr-vl'."
            )

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

        # Residual second-pass recovery (mask detected boxes -> re-OCR leftover
        # ink). "auto" (default) enables it for region-level recognizers, where
        # it's validated and do-no-harm-gated; it stays off for line recognizers
        # (e.g. Tesseract). True forces it on for any region-capable recognizer;
        # False disables it. See newspaper_ocr.residual_ocr.ResidualOcr.
        self.residual = None
        if residual_ocr:
            is_region = isinstance(self.recognizer, RegionRecognizer)
            enable = is_region if residual_ocr == "auto" else True
            if enable:
                if not (is_region or hasattr(self.recognizer, "recognize_region")):
                    raise ValueError(
                        "residual_ocr needs a region-capable recognizer (a "
                        "RegionRecognizer, or a line recognizer exposing "
                        f"recognize_region); got {type(self.recognizer).__name__}."
                    )
                from newspaper_ocr.residual_ocr import ResidualOcr
                self.residual = ResidualOcr(recognizer=self.recognizer)

    @staticmethod
    def _resolve_detector_name(detector: str) -> str:
        """Map the ``"auto"`` detector to the best available concrete detector.

        Preference order, best first: **DocLayout-YOLO**, then PaddleX
        (PP-DocLayout), then ``as_yolo``.  On NewsBench (dense, multi-column
        newspaper pages) DocLayout-YOLO + GLM-OCR leads PaddleX + GLM-OCR
        (0.970 vs 0.937 overall) — it proposes finer, more complete regions and
        needs no residual recovery — so ``"auto"`` prefers it when the
        ``doclayout-yolo`` package is installed, falls back to PaddleX when only
        that is present, and to ``as_yolo`` otherwise (with a warning, since it
        underperforms on broadsheets).  Any explicit name is returned unchanged
        (and resolves normally, erroring if unavailable).

        Install the preferred detector with ``newspaper-ocr[doclayout]`` (or
        ``[paddlex]``).
        """
        if detector != "auto":
            return detector
        import importlib.util

        if importlib.util.find_spec("doclayout_yolo") is not None:
            return "doclayout_yolo"
        if importlib.util.find_spec("paddlex") is not None:
            return "paddlex"
        import warnings

        warnings.warn(
            "Neither DocLayout-YOLO nor PaddleX is installed, so the 'auto' "
            "detector is falling back to 'as_yolo', which underperforms on dense "
            "newspaper pages. Install newspaper-ocr[doclayout] (best) or "
            "newspaper-ocr[paddlex] for substantially better layout detection.",
            stacklevel=3,
        )
        return "as_yolo"

    @staticmethod
    def _as_recognizer(recognizer):
        """Normalize a user-supplied recognizer into a recognizer object.

        Recognizer instances (or anything duck-typing ``recognize``) pass
        through untouched. A plain ``fn(image) -> text`` callable is wrapped in
        :class:`~newspaper_ocr.recognizers.custom.CallableRegionRecognizer` so
        folks can plug in their own OCR device — e.g. a call to OpenAI or
        OpenRouter — without subclassing.
        """
        if isinstance(recognizer, (LineRecognizer, RegionRecognizer)):
            return recognizer
        if hasattr(recognizer, "recognize") or hasattr(recognizer, "recognize_region"):
            return recognizer
        if callable(recognizer):
            from newspaper_ocr.recognizers.custom import CallableRegionRecognizer
            return CallableRegionRecognizer(recognizer)
        raise TypeError(
            "recognizer must be a name, a LineRecognizer/RegionRecognizer, or a "
            f"callable taking a PIL image and returning text; got {type(recognizer).__name__}"
        )

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
        texts are stitched back together. Status becomes ``ok`` only if every
        band was a clean read, ``chunked_partial`` if any band failed (timed out,
        errored, or looped) but others produced text, or ``timeout`` if nothing
        came back.
        """
        width, height = region.image.size
        spans = chunking.chunk_spans(height, self.chunk_height, self.chunk_overlap)

        texts: list[str] = []
        any_incomplete = False
        for y0, y1 in spans:
            band = Region(
                bbox=region.bbox,
                image=region.image.crop((0, y0, width, y1)),
                label=region.label,
            )
            band = self.recognizer.recognize(band)
            # Any non-clean band (timeout, error, repetition) means the merged
            # text may be missing or degraded content -> not a clean "ok".
            if band.status != "ok":
                any_incomplete = True
            # Skip the timeout placeholder; keep real text (incl. truncated).
            if band.status != "timeout" and band.text:
                texts.append(band.text)

        if not texts:
            return region  # keep the primary's timeout text/status
        region.text = chunking.merge_chunk_texts(texts)
        region.status = "chunked_partial" if any_incomplete else "ok"
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

    @staticmethod
    def _load_image(image: Image.Image | str | Path) -> Image.Image:
        """Normalize an input into an RGB :class:`PIL.Image.Image`.

        Accepts an open image or a path (loaded from disk), and converts any
        non-RGB mode — grayscale scans are common — so detectors and recognizers
        get a consistent 3-channel image. Anything else raises a clear
        ``TypeError`` instead of failing cryptically deep in a detector.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(str(image))
        if not isinstance(image, Image.Image):
            raise TypeError(
                "expected a PIL Image or a path to one, got "
                f"{type(image).__name__}"
            )
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def analyze(self, image: Image.Image | str | Path) -> PageLayout:
        """Detect, recognize and post-process a page, returning the layout.

        This is :meth:`run` without the formatting step, for callers that need
        the regions themselves — a review site, an article-segmentation pass, or
        anything that wants to emit more than one representation of a page
        without OCRing it twice.

        ``image`` may be an open :class:`PIL.Image.Image` or a path; non-RGB
        inputs (e.g. grayscale scans) are converted automatically.
        """
        image = self._load_image(image)
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

        # Residual second pass: recover text the detector never boxed (whole
        # columns/mastheads), then re-sort into reading order. Runs after the
        # recovery ladder so it works on the final pass-1 regions; gated to a
        # no-op when little ink is uncovered.
        if self.residual is not None:
            layout = self.residual.recover(layout)

        # Text cleaning (dehyphenation, line joining) only for line-level recognizers.
        # Region-level recognizers (GLM-OCR, VLMs) already return clean text.
        if isinstance(self.recognizer, LineRecognizer):
            layout = self.text_cleaner.clean(layout)

        layout = self.spell_checker.check(layout)
        return layout

    def run(self, image: Image.Image | str | Path) -> str:
        """Analyze a page and render it with the configured formatter."""
        return self.formatter.format(self.analyze(image))

    def ocr(self, path: str | Path, output: str | None = None) -> str:
        return self.run(path)

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
