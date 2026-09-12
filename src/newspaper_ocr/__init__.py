"""newspaper-ocr: Modular OCR pipeline for historical newspaper scans."""
from newspaper_ocr.errors import OcrTimeout
from newspaper_ocr.layout_processor import PIPELINE_REFERENCE_TAG
from newspaper_ocr.models import BBox, Line, Region, PageLayout, REGION_STATUSES
from newspaper_ocr.pipeline import Pipeline

__all__ = [
    "Pipeline",
    "BBox",
    "Line",
    "Region",
    "PageLayout",
    "REGION_STATUSES",
    "OcrTimeout",
    "PIPELINE_REFERENCE_TAG",
]
