"""newspaper-ocr: Modular OCR pipeline for historical newspaper scans."""
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from newspaper_ocr.pipeline import Pipeline

__all__ = ["Pipeline", "BBox", "Line", "Region", "PageLayout"]
