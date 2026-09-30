"""Hole-fill union of two layout detectors.

A *primary* detector supplies the base regions (and, if it has one, their
reading order).  A *secondary* detector contributes only "holes": boxes the
primary missed that contain real ink.  A secondary box is kept as a hole when

* less than ``overlap_max`` of its area is already covered — by the primary
  boxes, or by a hole accepted before it (candidates go in descending
  confidence, so overlapping secondary boxes don't produce duplicate holes); and
* at least ``ink_min`` of its pixels are ink, which screens out margins and
  whitespace-only granularity mismatches between the two detectors.

On NewsBench, MinerU2.5 boxes plus DocLayout-YOLO holes beat either detector
alone: MinerU's order is excellent but it misses isolated blocks (a poem stanza,
an ad, a side column) that DocLayout covers.  Most pages get few or no holes and
come through unchanged.  See issue #22.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.models import PageLayout, Region


class UnionDetector(Detector):
    """Primary detector's regions plus the secondary detector's holes.

    Parameters
    ----------
    primary, secondary : Detector
        The base detector and the hole-proposing detector.
    overlap_max : float
        Keep a secondary box only if less than this fraction of it is covered.
    ink_min : float
        Keep a secondary box only if at least this fraction of it is ink.
    ink_thresh : int or None
        Grayscale level below which a pixel counts as ink; ``None`` picks the
        level per page with Otsu's method (safer on dark microfilm).

    The returned layout tags each region's ``source`` as ``"primary"`` or
    ``"hole"``.  If the primary layout is ``ordered``, holes are inserted into
    its reading order (see :func:`~newspaper_ocr.layout_processor.insert_in_order`)
    and the result stays ``ordered``; otherwise holes are appended and layout
    post-processing sorts the page as usual.
    """

    def __init__(
        self,
        primary: Detector,
        secondary: Detector,
        overlap_max: float = 0.15,
        ink_min: float = 0.012,
        ink_thresh: int | None = 128,
    ):
        self.primary = primary
        self.secondary = secondary
        self.overlap_max = overlap_max
        self.ink_min = ink_min
        self.ink_thresh = ink_thresh
        #: Number of holes added on the most recent page, for diagnostics.
        self.last_n_holes = 0

    def detect(self, image: Image.Image) -> PageLayout:
        base = self.primary.detect(image)
        cand = self.secondary.detect(image)
        holes = self.find_holes(image, base.regions, cand.regions)
        self.last_n_holes = len(holes)

        for r in base.regions:
            r.source = "primary"
        for r in holes:
            r.source = "hole"

        if base.ordered:
            from newspaper_ocr.layout_processor import insert_in_order
            regions = insert_in_order(base.regions, holes)
        else:
            regions = base.regions + holes

        return PageLayout(
            image=image,
            regions=regions,
            width=base.width or image.size[0],
            height=base.height or image.size[1],
            # Only claim line detection if every region could have lines.
            lines_detected=base.lines_detected and cand.lines_detected,
            ordered=base.ordered,
        )

    def find_holes(
        self,
        image: Image.Image,
        primary: list[Region],
        candidates: list[Region],
    ) -> list[Region]:
        """Return the candidates that fill holes in *primary* coverage."""
        ink = self._ink_mask(image)
        h, w = ink.shape
        covered = np.zeros((h, w), dtype=bool)
        for r in primary:
            covered[self._slice(r, w, h)] = True

        holes: list[Region] = []
        for r in sorted(candidates, key=lambda r: -r.confidence):
            sl = self._slice(r, w, h)
            if covered[sl].size == 0:
                continue
            if covered[sl].mean() >= self.overlap_max:
                continue
            if ink[sl].mean() < self.ink_min:
                continue
            holes.append(r)
            covered[sl] = True
        return holes

    def _ink_mask(self, image: Image.Image) -> np.ndarray:
        gray = np.asarray(image.convert("L"))
        if self.ink_thresh is not None:
            return gray < self.ink_thresh
        import cv2
        _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binv.astype(bool)

    @staticmethod
    def _slice(r: Region, w: int, h: int) -> tuple[slice, slice]:
        b = r.bbox
        return (slice(max(0, b.y0), min(h, b.y1)), slice(max(0, b.x0), min(w, b.x1)))
