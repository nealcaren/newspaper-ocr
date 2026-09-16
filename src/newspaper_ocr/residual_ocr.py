"""Residual second-pass OCR to recover text the detector never boxed.

With a strong region recognizer (e.g. GLM-OCR) and a good detector (PP-DocLayout)
precision is saturated; the remaining error is *recall* — ordinary prose in whole
blocks (mastheads, side columns, inter-block strips) that the detector never
proposed as a region, so the recognizer never saw it.  ``LayoutProcessor`` and
``RegionRepair`` cannot fix this: they only operate *within* the boxes the
detector already produced.

This pass works on the ink the boxes missed.  It runs AFTER recognition and is
**non-destructive**: it returns a new :class:`PageLayout`; the input is untouched
(same contract as :class:`~newspaper_ocr.region_repair.RegionRepair`).

Algorithm (per pass)
--------------------
1. **Ink mask** — threshold the page (Otsu by default) so dark pixels = ink.
2. **Gate** — paint every current region box (padded) as "covered"; measure the
   fraction of ink that lies *outside* every box.  If little ink is uncovered the
   detector already saw ~everything, so the pass is a no-op (do-no-harm on clean
   pages).
3. **Block-ify (recursive XY-cut)** — a detector miss is usually a whole
   *multi-column* area, which a blind connected-components pass either shatters
   into per-line fragments or fuses into one unreadable page-tall blob.  Instead
   we cut the uncovered ink the way a reader parses a page: split at wide vertical
   gutters (columns), then at tall horizontal gaps (blocks within a column),
   recursively, until neither a column gutter nor a block gap remains.  This
   yields a handful of column-shaped blocks in the right shape and reading order.
4. **Recognize** each residual block with the region recognizer (column-sized
   crops — no whole-page blowup, no per-line call storm).
5. **Merge + re-sort** — add the residual regions to the pass-1 set and re-run the
   existing reading-order sort so they land in sequence, not appended.

Because covered ink is erased before step 3, recovered crops **cannot** duplicate
already-captured text, so no lossy dedup is needed and precision is preserved.

The pass repeats until no meaningful uncovered ink remains (a natural convergence
criterion), bounded by ``max_passes``.
"""
from __future__ import annotations

import cv2
import numpy as np

from newspaper_ocr.models import BBox, PageLayout, Region
from newspaper_ocr.recognizers.base import RegionRecognizer


class ResidualOcr:
    """Recover detector-missed text via a mask-and-re-OCR second pass.

    Parameters
    ----------
    recognizer:
        Region-level recognizer used to read each residual crop.  Required — this
        pass exists to OCR blocks the primary run never saw.
    pad:
        Pixels each pass-1 box is grown by before it is treated as "covered", so a
        box's own faint edge ink doesn't reappear as a residual sliver.
    All geometry knobs below (``min_block_*``, ``min_gutter``, ``min_row_gap``,
    ``max_block_w``) default to ``None``, meaning "derive from this page's
    measured column width" (see :meth:`_resolve_scale` and :data:`_SCALE_FRAC`),
    so the pass adapts to a page's DPI and column count.  Pass an int to pin one.

    min_block_area, min_block_w, min_block_h:
        A residual block smaller than any of these (px / px²) is discarded as a
        speck.  ``None`` -> ~0.06·colW (w/h) and ~0.018·colW² (area).
    min_gutter:
        Minimum width (px) of a near-empty vertical band for the XY-cut to treat
        it as a column gutter and split there.  Below a real gutter, above the
        inter-word gaps inside a column.
    min_row_gap:
        Minimum height (px) of a near-empty horizontal band for the XY-cut to
        split a column into separate blocks.  Above inter-line spacing, so lines
        of a paragraph stay together; around major article/paragraph breaks.
    max_block_w:
        A block wider than this is assumed to span more than one column and is
        force-split at the deepest valley in its vertical ink projection, even
        when a horizontal rule keeps the gutter from projecting fully empty (the
        classic XY-cut failure).  Set near a single column's width.
    proj_frac_eps:
        A projection row/column counts as "empty" when its ink is below this
        fraction of the sub-region's cross length — makes gutter/gap detection
        robust to speckle instead of requiring exactly zero ink.
    ink_thresh:
        Fixed 0-255 threshold for the ink mask, or ``None`` for Otsu (default).
    run_if_uncovered_ink:
        Gate.  The pass runs only when at least this fraction of the page's ink
        lies outside every current box; otherwise it returns the layout unchanged.
    max_passes:
        Convergence cap on the mask -> recover -> re-mask loop.
    min_gain_ink:
        Stop early when a pass leaves less than this fraction of page ink still
        uncovered improvement — i.e. it recovered almost nothing new.
    label:
        Label assigned to recovered regions (default ``"text"``).
    """

    #: Geometry knobs, as a fraction of the page's measured column width, used
    #: whenever the corresponding constructor argument is left ``None``.  The
    #: values are the hand-tuned pixel constants divided by the ~665px column
    #: width they were tuned on, so a page at any DPI or column count gets the
    #: same *relative* geometry.  ``min_block_area`` is a fraction of colW².
    _SCALE_FRAC = {
        "min_gutter": 0.045,
        "min_row_gap": 0.090,
        "max_block_w": 1.40,
        "min_block_w": 0.060,
        "min_block_h": 0.060,
    }
    _AREA_FRAC = 0.018          # min_block_area / colW²
    _COLW_FALLBACK = 665        # px, if a page has too few boxes to measure columns

    def __init__(
        self,
        recognizer: RegionRecognizer,
        *,
        pad: int = 6,
        min_block_area: int | None = None,
        min_block_w: int | None = None,
        min_block_h: int | None = None,
        min_gutter: int | None = None,
        min_row_gap: int | None = None,
        max_block_w: int | None = None,
        proj_frac_eps: float = 0.01,
        max_cut_depth: int = 16,
        ink_thresh: int | None = None,
        run_if_uncovered_ink: float = 0.10,
        max_passes: int = 2,
        min_gain_ink: float = 0.02,
        label: str = "text",
    ) -> None:
        if recognizer is None:
            raise ValueError("ResidualOcr needs a region-level recognizer to read "
                             "the residual crops it finds.")
        self.recognizer = recognizer
        self.pad = pad
        # Geometry knobs: an explicit value pins it; ``None`` means derive it from
        # the page's own column width in :meth:`_resolve_scale` per ``recover`` call.
        self._overrides = {
            "min_block_area": min_block_area,
            "min_block_w": min_block_w,
            "min_block_h": min_block_h,
            "min_gutter": min_gutter,
            "min_row_gap": min_row_gap,
            "max_block_w": max_block_w,
        }
        # Concrete starting values (used by direct _blocks calls before recover
        # resolves them, and as the resolved value for any pinned knob).
        for name, frac in self._SCALE_FRAC.items():
            setattr(self, name, self._overrides[name]
                    if self._overrides[name] is not None
                    else round(frac * self._COLW_FALLBACK))
        self.min_block_area = (min_block_area if min_block_area is not None
                               else round(self._AREA_FRAC * self._COLW_FALLBACK ** 2))
        self.proj_frac_eps = proj_frac_eps
        self.max_cut_depth = max_cut_depth
        self.ink_thresh = ink_thresh
        self.run_if_uncovered_ink = run_if_uncovered_ink
        self.max_passes = max_passes
        self.min_gain_ink = min_gain_ink
        self.label = label
        self.colW: float = float(self._COLW_FALLBACK)  # last measured column width
        self.actions: list[tuple] = []      # (kind, ...) log of what was done

    # -- per-page scale ---------------------------------------------------------
    def _resolve_scale(self, regions: list[Region]) -> None:
        """Set each un-pinned geometry knob from this page's column width.

        The column width is the single measured scale everything derives from
        (see :data:`_SCALE_FRAC`).  It is read from the detected boxes via
        ``LayoutProcessor._find_columns`` — the same routine reading-order uses —
        so the residual pass adapts to a page's DPI and column count instead of
        assuming the ~5000px broadsheets it was tuned on.
        """
        import numpy as np
        from newspaper_ocr.layout_processor import _bbox_tuple, _find_columns

        colW = float(self._COLW_FALLBACK)
        if len(regions) >= 3:
            boxes = np.asarray([_bbox_tuple(r) for r in regions], dtype=int)
            cols, median_w = _find_columns(boxes)
            widths = [cr - cl for _, cl, cr in cols]
            if widths:
                colW = float(np.median(widths))
            elif median_w > 0:
                colW = float(median_w)
        self.colW = colW

        for name, frac in self._SCALE_FRAC.items():
            if self._overrides[name] is None:
                setattr(self, name, max(1, round(frac * colW)))
        if self._overrides["min_block_area"] is None:
            self.min_block_area = round(self._AREA_FRAC * colW ** 2)

    # -- public entry -----------------------------------------------------------
    def recover(self, layout: PageLayout) -> PageLayout:
        """Run the residual pass(es) and return a new :class:`PageLayout`."""
        self.actions = []
        regions = list(layout.regions)
        self._resolve_scale(regions)
        ink = self._ink_mask(layout.image)
        total_ink = int(ink.sum())
        if total_ink == 0:
            return self._rebuild(layout, regions)

        for _ in range(self.max_passes):
            covered = self._covered_mask(ink.shape, regions)
            uncovered = ink & ~covered
            frac = uncovered.sum() / total_ink
            if frac < self.run_if_uncovered_ink:
                self.actions.append(("skip", round(float(frac), 4)))
                break

            blocks = self._blocks(uncovered)
            if not blocks:
                self.actions.append(("no_blocks", round(float(frac), 4)))
                break

            recovered = self._recognize_blocks(layout.image, blocks)
            self.actions.append(("pass", len(recovered), round(float(frac), 4)))
            if not recovered:
                break
            regions = regions + recovered

            # Convergence: if what these blocks covered is a tiny slice of page
            # ink, another pass won't find much either.
            gained = self._covered_mask(ink.shape, recovered) & ink
            if gained.sum() / total_ink < self.min_gain_ink:
                break

        regions = self._reading_order(regions, layout)
        return self._rebuild(layout, regions)

    # -- ink / coverage masks ---------------------------------------------------
    def _ink_mask(self, page_img) -> np.ndarray:
        """Boolean mask (H, W): True where the page has ink (dark pixels)."""
        gray = cv2.cvtColor(np.asarray(page_img), cv2.COLOR_RGB2GRAY)
        if self.ink_thresh is None:
            _, binv = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binv = cv2.threshold(gray, self.ink_thresh, 255, cv2.THRESH_BINARY_INV)
        return binv.astype(bool)

    def _covered_mask(self, shape: tuple[int, int],
                      regions: list[Region]) -> np.ndarray:
        """Boolean mask: True inside any region box (grown by ``pad``)."""
        h, w = shape
        covered = np.zeros((h, w), dtype=bool)
        for r in regions:
            b = r.bbox
            x0 = max(0, b.x0 - self.pad)
            y0 = max(0, b.y0 - self.pad)
            x1 = min(w, b.x1 + self.pad)
            y1 = min(h, b.y1 + self.pad)
            if x1 > x0 and y1 > y0:
                covered[y0:y1, x0:x1] = True
        return covered

    # -- residual blocks (recursive XY-cut) -------------------------------------
    @staticmethod
    def _gap_segments(proj: np.ndarray, eps: float, min_gap: int) -> list[tuple[int, int]]:
        """Split a 1-D ink projection into content runs at empty gaps.

        A position is "empty" when its ink is ``<= eps``.  Runs of empty
        positions at least ``min_gap`` long are treated as separators; content
        segments between them are returned as ``(start, end)`` index pairs, with
        leading/trailing empties trimmed.  Returns a single segment when no gap is
        wide enough to split on.
        """
        empty = proj <= eps
        segments: list[tuple[int, int]] = []
        n = len(proj)
        i = 0
        while i < n:
            if empty[i]:
                i += 1
                continue
            j = i
            while j < n:
                if empty[j]:
                    # look ahead: is this a wide-enough gap to end the segment?
                    k = j
                    while k < n and empty[k]:
                        k += 1
                    if k - j >= min_gap or k == n:
                        break
                    j = k            # narrow gap — stay in the same segment
                else:
                    j += 1
            segments.append((i, j))
            i = j
        return segments

    def _blocks(self, uncovered: np.ndarray) -> list[BBox]:
        """Recursive XY-cut of the uncovered-ink mask into column-shaped blocks."""
        boxes: list[BBox] = []
        # stack items: (x0, y0, x1, y1, depth, cut_axis) — cut_axis alternates so a
        # region that just failed a vertical cut is next tried horizontally.
        stack: list[tuple[int, int, int, int, int, str]] = [
            (0, 0, uncovered.shape[1], uncovered.shape[0], 0, "v")
        ]
        while stack:
            x0, y0, x1, y1, depth, axis = stack.pop()
            sub = uncovered[y0:y1, x0:x1]
            if sub.shape[0] < self.min_block_h or sub.shape[1] < self.min_block_w:
                continue

            split = False
            if depth < self.max_cut_depth:
                if axis == "v":
                    proj = sub.sum(axis=0)          # ink per column
                    eps = self.proj_frac_eps * sub.shape[0]
                    segs = self._gap_segments(proj, eps, self.min_gutter)
                    if len(segs) > 1:
                        for a, b in segs:
                            stack.append((x0 + a, y0, x0 + b, y1, depth + 1, "h"))
                        split = True
                else:
                    proj = sub.sum(axis=1)          # ink per row
                    eps = self.proj_frac_eps * sub.shape[1]
                    segs = self._gap_segments(proj, eps, self.min_row_gap)
                    if len(segs) > 1:
                        for a, b in segs:
                            stack.append((x0, y0 + a, x1, y0 + b, depth + 1, "v"))
                        split = True
            if split:
                continue
            # No cut on this axis. Try the other axis once before emitting a leaf.
            if axis == "v" and depth < self.max_cut_depth:
                stack.append((x0, y0, x1, y1, depth + 1, "h"))
                continue

            # Both gap-cuts failed. A block wider than one column is a multi-column
            # blob a spanning rule kept the gutter cut from finding — force-split it
            # at the deepest valley in its vertical projection.
            if (x1 - x0) > self.max_block_w and depth < self.max_cut_depth:
                cut = self._valley_cut(uncovered[y0:y1, x0:x1])
                if cut is not None:
                    stack.append((x0, y0, x0 + cut, y1, depth + 1, "h"))
                    stack.append((x0 + cut, y0, x1, y1, depth + 1, "h"))
                    continue

            box = self._tighten(uncovered, x0, y0, x1, y1)
            if box is not None:
                boxes.append(box)
        return boxes

    def _valley_cut(self, sub: np.ndarray) -> int | None:
        """Local-x index of the lowest-ink column in the central span, or None.

        Restricts the search to the middle of the block so the cut lands near a
        real column gutter, not at a sparse edge, and only returns a cut that
        leaves both sides at least ``min_block_w`` wide.
        """
        w = sub.shape[1]
        lo, hi = int(w * 0.2), int(w * 0.8)
        if hi - lo < 1:
            return None
        proj = sub.sum(axis=0)
        cut = lo + int(np.argmin(proj[lo:hi]))
        if cut < self.min_block_w or w - cut < self.min_block_w:
            return None
        return cut

    def _tighten(self, uncovered: np.ndarray, x0: int, y0: int, x1: int,
                 y1: int) -> BBox | None:
        """Shrink a leaf rectangle to the bounding box of its ink, then size-filter."""
        sub = uncovered[y0:y1, x0:x1]
        rows = np.where(sub.any(axis=1))[0]
        cols = np.where(sub.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return None
        bx0, bx1 = x0 + int(cols[0]), x0 + int(cols[-1]) + 1
        by0, by1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
        w, h = bx1 - bx0, by1 - by0
        if w < self.min_block_w or h < self.min_block_h or w * h < self.min_block_area:
            return None
        return BBox(bx0, by0, bx1, by1)

    def _recognize_blocks(self, page_img, blocks: list[BBox]) -> list[Region]:
        """Re-OCR each residual block; keep only the ones that yield text.

        A residual block is a whole region (a column strip), so it is OCR'd with
        ``recognize_region`` when the recognizer offers it (line recognizers such
        as Tesseract) and with ``recognize`` otherwise (region-level VLMs like
        GLM-OCR).  This lets the pass improve line- and region-level engines alike.
        """
        region_ocr = getattr(self.recognizer, "recognize_region", None)
        out: list[Region] = []
        for b in blocks:
            crop = page_img.crop(b.to_tuple())
            region = Region(bbox=b, image=crop, label=self.label)
            r = region_ocr(region) if callable(region_ocr) else self.recognizer.recognize(region)
            if (r.text or "").strip():
                r.engine = "residual"
                out.append(r)
        return out

    # -- merge + re-sort --------------------------------------------------------
    @staticmethod
    def _reading_order(regions: list[Region], layout: PageLayout) -> list[Region]:
        """Interleave recovered regions into newspaper reading order."""
        from newspaper_ocr.layout_processor import LayoutProcessor
        return LayoutProcessor()._reading_order(regions)

    @staticmethod
    def _rebuild(layout: PageLayout, regions: list[Region]) -> PageLayout:
        for i, r in enumerate(regions):
            r.id = f"r{i}"
        return PageLayout(image=layout.image, regions=regions,
                          width=layout.width, height=layout.height,
                          lines_detected=layout.lines_detected)
