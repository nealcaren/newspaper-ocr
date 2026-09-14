"""Core data models for the OCR pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Line:
    bbox: BBox
    image: Image.Image
    text: str = ""
    confidence: float = 0.0


# Per-region OCR outcomes recorded in Region.status.
REGION_STATUSES = ("ok", "timeout", "repetition", "error")

#: Placeholder text written when a region exhausts its retries on a timeout.
#: Lives here rather than in one recognizer because every consumer that reads
#: ``Region.text`` needs to tell it apart from recognized text — a repair or
#: dedup pass that treats it as content will happily "deduplicate" two
#: unrelated regions that both timed out.
TIMEOUT_TEXT = "[OCR timeout]"


@dataclass
class Region:
    """A detected page region and its recognized text.

    Attributes
    ----------
    id : str
        Stable identifier within the page, assigned by the pipeline as ``r0``,
        ``r1``, ... in reading order.  Downstream consumers (review sites, LLM
        enrichment passes) use it to refer back to a region.
    status : str
        Outcome of recognition for this region — one of :data:`REGION_STATUSES`.
        ``"ok"`` is a clean read; ``"timeout"`` means the recognizer exceeded its
        wall-clock budget; ``"repetition"`` means the model looped and the text
        was truncated; ``"error"`` means recognition raised.  Anything other than
        ``"ok"`` marks the region as a candidate for a re-OCR pass.
    """

    bbox: BBox
    image: Image.Image
    label: str
    lines: list[Line] = field(default_factory=list)
    text: str = ""
    confidence: float = 0.0
    status: str = "ok"
    id: str = ""


@dataclass
class PageLayout:
    """A detected page: its regions, in reading order once processed.

    Attributes
    ----------
    lines_detected : bool
        Whether a line detector ran over this page.  Layout post-processing uses
        it to tell "the line detector found nothing here" (a text region that is
        probably a false positive) from "nobody looked" (a region-only detector,
        or line detection skipped for speed).  Defaults to False so a detector
        has to opt in: the cost of not setting it is a false positive surviving,
        while the cost of wrongly setting it is deleting real text.
    raw_regions : list[Region] | None
        The regions as recognition left them, before any post-recognition
        repair (:mod:`newspaper_ocr.region_repair`).  ``None`` until a repair
        pass runs; from then on it is the untouched layer repair always
        recomputes *from*, never overwrites.  That is what makes repair
        idempotent and re-runnable with different thresholds: the raw OCR
        survives a repair whose logic turns out to be wrong.
    """

    image: Image.Image
    regions: list[Region] = field(default_factory=list)
    width: int = 0
    height: int = 0
    lines_detected: bool = False
    raw_regions: list[Region] | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(r.text for r in self.regions if r.text)
