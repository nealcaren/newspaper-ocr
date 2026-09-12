import json
from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.models import PageLayout


class JsonFormatter(Formatter):
    """Page JSON: the stable contract for downstream consumers.

    Each region carries ``id``, ``label``, ``bbox``, ``text``, ``status`` and
    ``confidence``.  ``status`` (see :data:`newspaper_ocr.models.REGION_STATUSES`)
    lets a caller find regions worth re-OCRing without re-reading the images, and
    ``id`` gives article-segmentation or review-site passes a stable handle.
    """

    def format(self, layout: PageLayout) -> str:
        data = {
            "width": layout.width,
            "height": layout.height,
            "regions": [
                {
                    "id": r.id or f"r{i}",
                    "label": r.label,
                    "bbox": {"x0": r.bbox.x0, "y0": r.bbox.y0, "x1": r.bbox.x1, "y1": r.bbox.y1},
                    "text": r.text,
                    "status": r.status,
                    "confidence": r.confidence,
                    "lines": [
                        {
                            "text": line.text,
                            "confidence": line.confidence,
                            "bbox": {"x0": line.bbox.x0, "y0": line.bbox.y0, "x1": line.bbox.x1, "y1": line.bbox.y1},
                        }
                        for line in r.lines
                    ],
                }
                for i, r in enumerate(layout.regions)
            ],
        }
        return json.dumps(data, indent=2)
