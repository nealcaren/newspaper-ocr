from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.models import BBox, PageLayout, Region


class PaddleXDetector(Detector):
    """PP-DocLayout_plus-L detector. Requires: pip install paddlepaddle paddlex"""

    def __init__(self, model_name: str = "PP-DocLayout_plus-L", **kwargs):
        try:
            import os

            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            from paddlex import create_model
        except ImportError:
            raise ImportError(
                "PaddleX is not installed. Install with:\n"
                "  pip install paddlepaddle paddlex\n"
                "  or: uv pip install paddlepaddle paddlex"
            )
        self.model = create_model(model_name)

    def detect(self, image: Image.Image) -> PageLayout:
        w, h = image.size

        # PaddleX needs a file path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f.name)
            results = list(self.model.predict(f.name))
            Path(f.name).unlink(missing_ok=True)

        regions = []
        for result in results:
            for box in result.get("boxes", []):
                coords = box.get("coordinate", [0, 0, 0, 0])
                x0, y0, x1, y1 = [int(c) for c in coords]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 <= x0 or y1 <= y0:
                    continue

                crop = image.crop((x0, y0, x1, y1))
                regions.append(
                    Region(
                        bbox=BBox(x0, y0, x1, y1),
                        image=crop,
                        label=box.get("label", "unknown"),
                        lines=[],
                        confidence=box.get("score", 0.0),
                    )
                )

        return PageLayout(image=image, regions=regions, width=w, height=h)
