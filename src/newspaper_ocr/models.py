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
#   ok              — clean read
#   timeout         — recognizer exceeded its wall-clock budget
#   repetition      — model looped; text was truncated
#   error           — recognition raised
#   chunked_partial — a tall region was split into vertical chunks and at least
#                     one chunk timed out, so the merged text is real but incomplete
#   hallucination   — the read was in a script the page doesn't use (CJK on an
#                     English page) and was blanked; see script_filter
REGION_STATUSES = ("ok", "timeout", "repetition", "error", "chunked_partial",
                   "hallucination")


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
    #: When a fallback recognizer replaces the primary text (do-no-harm
    #: recovery), the primary's original text is kept here so the swap is
    #: reversible and auditable.
    text_primary: str = ""
    #: Which engine produced the final text, when it wasn't the primary
    #: recognizer — e.g. the fallback recognizer's class name. Empty means the
    #: primary recognizer's result was kept.
    engine: str = ""
    #: Which detector proposed the region when detectors are combined (see
    #: :class:`~newspaper_ocr.detectors.union.UnionDetector`): ``"primary"`` or
    #: ``"hole"``.  Empty for a single detector.
    source: str = ""


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
    ordered : bool
        Whether ``regions`` already arrive in reading order from the detector
        (e.g. a VLM layout model with native ordering).  Layout post-processing
        and the residual pass then keep that order — inserting any new regions
        into it — instead of re-sorting, merging or gap-filling the page.
    alternates : list[Region]
        A second detector's boxes that were not used as regions (the hole-fill
        detector's leftovers).  The pipeline reads them in place of a large
        region whose read came back empty; see ``Pipeline(rescue_empty_reads=)``.
    """

    image: Image.Image
    regions: list[Region] = field(default_factory=list)
    width: int = 0
    height: int = 0
    lines_detected: bool = False
    ordered: bool = False
    alternates: list[Region] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(r.text for r in self.regions if r.text)
