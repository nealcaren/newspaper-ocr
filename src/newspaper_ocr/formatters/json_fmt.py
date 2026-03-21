import json
from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.models import PageLayout


class JsonFormatter(Formatter):
    def format(self, layout: PageLayout) -> str:
        data = {
            "width": layout.width,
            "height": layout.height,
            "regions": [
                {
                    "label": r.label,
                    "bbox": {"x0": r.bbox.x0, "y0": r.bbox.y0, "x1": r.bbox.x1, "y1": r.bbox.y1},
                    "text": r.text,
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
                for r in layout.regions
            ],
        }
        return json.dumps(data, indent=2)
