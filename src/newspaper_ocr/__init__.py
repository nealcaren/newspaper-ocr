"""newspaper-ocr: Modular OCR pipeline for historical newspaper scans."""
from newspaper_ocr.errors import OcrTimeout
from newspaper_ocr.layout_processor import PIPELINE_REFERENCE_TAG
from newspaper_ocr.models import (
    BBox,
    Line,
    Region,
    PageLayout,
    REGION_STATUSES,
    TIMEOUT_TEXT,
)
from newspaper_ocr.pipeline import Pipeline
from newspaper_ocr.region_repair import (
    DUPLICATE_PAGE_THRESHOLD,
    DuplicatePage,
    RegionRepair,
    find_duplicate_pages,
    page_similarity,
)

__all__ = [
    "Pipeline",
    "BBox",
    "Line",
    "Region",
    "PageLayout",
    "REGION_STATUSES",
    "TIMEOUT_TEXT",
    "RegionRepair",
    "DuplicatePage",
    "find_duplicate_pages",
    "page_similarity",
    "DUPLICATE_PAGE_THRESHOLD",
    "OcrTimeout",
    "PIPELINE_REFERENCE_TAG",
]
