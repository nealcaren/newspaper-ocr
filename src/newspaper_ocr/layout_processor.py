"""Layout post-processing for newspaper OCR pages.

Ported from the dangerouspress-ocr production pipeline (ocr_pipeline.py).
Adapts dict-based logic to use the Region / PageLayout data model.

The port tracks a specific revision of that pipeline, recorded in
:data:`PIPELINE_REFERENCE_TAG`.  The tuned constants below are the ones that
must stay in sync with it: ``gap_thresh = median_w * 0.3`` when splitting
columns, the 40%-of-median narrow-column merge, the ``max_height=600`` cap on
merging adjacent blocks, and the 0.5 / 0.15 confidence bands.  If ocr_pipeline.py
moves past that tag, diff those values first — a silent drift here changes
column segmentation, and therefore the text, for every page.

Pipeline stages (in order):
  1. _filter          – drop regions below confidence threshold
  2. _rescue_low_confidence – re-admit low-conf regions that don't overlap accepted ones
  3. _deduplicate      – remove overlapping / contained duplicates
  4. _fill_column_gaps – add synthetic text regions for large vertical gaps in columns
  5. _reading_order    – sort regions in newspaper column order (top-to-bottom per column)
  6. _merge_adjacent   – merge vertically adjacent same-column text blocks
  7. _drop_empty_overlaps – drop OCR-label regions the line detector found
     nothing in (skipped when no line detection ran — see PageLayout.lines_detected)

When the detector already supplies reading order (``PageLayout.ordered``), only
the confidence filter/rescue and the empty-overlap drop run, in place: sorting,
merging, gap-filling and dedup would all second-guess the detector's own blocks.
New regions (hole fill, residual recovery) go in with :func:`insert_in_order`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from newspaper_ocr.models import BBox, PageLayout, Region

#: Revision of dangerouspress-ocr/ocr_pipeline.py this module was ported from.
PIPELINE_REFERENCE_TAG = "2025-03-07-col-fix"

# Labels treated as "text content" regions.
_OCR_LABELS = {"text", "paragraph_title", "doc_title", "figure_title"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bbox_tuple(r: Region) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) from a Region."""
    return (r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1)


def _box_area(b: tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _intersection_area(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _find_columns(
    boxes: np.ndarray,
) -> tuple[list[tuple[float, float, float]], float]:
    """Find column boundaries from narrow (single-column-width) elements.

    Uses 1-D X-midpoint clustering.  Returns (col_centers, median_width) where
    each entry in col_centers is (center_x, left_x, right_x).
    """
    widths = boxes[:, 2] - boxes[:, 0]
    median_w = float(np.median(widths))
    narrow_mask = widths <= median_w * 1.3
    if np.sum(narrow_mask) < 3:
        return [], median_w

    narrow_mids = (boxes[narrow_mask, 0] + boxes[narrow_mask, 2]) / 2.0
    sorted_mids = np.sort(narrow_mids)
    gap_thresh = median_w * 0.3
    diffs = sorted_mids[1:] - sorted_mids[:-1]
    split_points = sorted_mids[:-1][diffs > gap_thresh] + diffs[diffs > gap_thresh] / 2

    labels = np.zeros(len(narrow_mids), dtype=int)
    for sp in split_points:
        labels[narrow_mids > sp] += 1

    col_centers: list[tuple[float, float, float]] = []
    for c in range(int(labels.max()) + 1):
        members = narrow_mids[labels == c]
        if len(members) > 0:
            col_centers.append(
                (
                    float(np.mean(members)),
                    float(np.min(boxes[narrow_mask][labels == c, 0])),
                    float(np.max(boxes[narrow_mask][labels == c, 2])),
                )
            )
    col_centers.sort(key=lambda x: x[0])

    # Merge columns narrower than 40 % of median_w into nearest neighbour
    min_col_w = median_w * 0.4
    filtered: list[tuple[float, float, float]] = []
    for center, cl, cr in col_centers:
        if cr - cl >= min_col_w:
            filtered.append((center, cl, cr))
        elif filtered:
            pc, pcl, pcr = filtered[-1]
            filtered[-1] = (pc, pcl, max(pcr, cr))
    return filtered, median_w


def _overlapping_cols(
    x1: float, x2: float, col_centers: list[tuple[float, float, float]]
) -> list[int]:
    """Columns a box spanning x1..x2 covers (>20% of the column's width), or
    the nearest column by centre when it covers none."""
    cols = []
    for c, (center, cl, cr) in enumerate(col_centers):
        overlap = min(x2, cr) - max(x1, cl)
        col_w = cr - cl
        if col_w > 0 and overlap > col_w * 0.2:
            cols.append(c)
    return cols if cols else [
        min(range(len(col_centers)),
            key=lambda c: abs(col_centers[c][0] - (x1 + x2) / 2))
    ]


def _x_overlaps(a: Region, b: Region, frac: float = 0.2) -> bool:
    """Whether a and b share more than *frac* of the narrower one's width."""
    overlap = min(a.bbox.x1, b.bbox.x1) - max(a.bbox.x0, b.bbox.x0)
    return overlap > frac * min(a.bbox.width, b.bbox.width)


def _is_above(b: Region, e: Region, frac: float = 0.3) -> bool:
    """Whether b sits above e, allowing a small vertical overlap."""
    tol = frac * min(b.bbox.height, e.bbox.height)
    return b.bbox.y1 <= e.bbox.y0 + tol


def _left_in_row(b: Region, e: Region, frac: float = 0.3) -> bool:
    """Whether b sits to the left of e and shares its row (vertical overlap)."""
    v_overlap = min(b.bbox.y1, e.bbox.y1) - max(b.bbox.y0, e.bbox.y0)
    tol = frac * min(b.bbox.width, e.bbox.width)
    return (b.bbox.x1 <= e.bbox.x0 + tol
            and v_overlap > frac * min(b.bbox.height, e.bbox.height))


def insert_in_order(base: list[Region], extras: list[Region]) -> list[Region]:
    """Insert *extras* into an already-ordered *base* without re-sorting it.

    Each extra is placed by its neighbours in the base order, not by a global
    column sweep, so it works for sectioned pages (a mid-page banner starting
    a new set of columns) and for row-major orders alike:

    1. Of the base regions above it in its horizontal span, take the one
       latest in reading order and insert right after it — unless the region
       directly above is much wider (a masthead or banner opening a section),
       which doesn't count as a predecessor.  When the base order runs row by
       row rather than column by column, the candidates are instead every
       region above it on the page plus those to its left in its row.
    2. With nothing above it in its span and nothing to its left in its row
       (the top-left of the page, or a masthead), it goes first.
    3. Otherwise insert before the earliest-ordered base region below it in
       its span.
    4. With no base region in its span at all, fall back to a column sweep:
       before the first base region in a later column, or lower in the same
       column.

    Extras that land at the same point keep column, then top-to-bottom, then
    left-to-right order among themselves.
    """
    if not extras:
        return list(base)
    if not base:
        return sorted(extras, key=lambda r: (r.bbox.y0, r.bbox.x0))

    boxes = np.asarray([_bbox_tuple(r) for r in base + extras], dtype=int)
    col_centers, _ = _find_columns(boxes)

    def col(r: Region) -> int:
        if not col_centers:
            return 0
        return min(_overlapping_cols(r.bbox.x0, r.bbox.x1, col_centers))

    base_cols = [col(b) for b in base]
    # Row-major if the order more often jumps back left to start a new row
    # than back up to start a new column.
    pairs = list(zip(base, base[1:]))
    new_rows = sum(_is_above(a, b) and b.bbox.x1 <= a.bbox.x0 for a, b in pairs)
    new_cols = sum(_is_above(b, a) and b.bbox.x0 >= a.bbox.x1 for a, b in pairs)
    row_major = new_rows > new_cols

    def position(e: Region, ec: int) -> int:
        in_span = [i for i, b in enumerate(base) if _x_overlaps(b, e)]
        above = [i for i in in_span if _is_above(base[i], e)]
        left = [i for i, b in enumerate(base) if _left_in_row(b, e)]
        # A much wider region directly above (masthead, banner, headline) opens
        # a section rather than preceding e in its column; place e by what
        # follows it instead.
        under_wide = bool(above) and (
            base[max(above, key=lambda i: base[i].bbox.y1)].bbox.width
            > 1.5 * e.bbox.width
        )
        if row_major:
            # Row by row: everything in earlier rows, plus this row's left side.
            before = [i for i, b in enumerate(base) if _is_above(b, e)] + left
        else:
            before = [] if under_wide else above
        if before:
            return max(before) + 1
        if not above and not left:
            return 0
        below = [i for i in in_span if base[i].bbox.y0 >= e.bbox.y0]
        if below:
            return min(below)
        for i, b in enumerate(base):
            if base_cols[i] > ec or (base_cols[i] == ec and b.bbox.y0 > e.bbox.y0):
                return i
        return len(base)

    placed: list[tuple[int, int, int, int, int, Region]] = []
    for k, e in enumerate(extras):
        ec = col(e)
        placed.append((position(e, ec), ec, e.bbox.y0, e.bbox.x0, k, e))
    placed.sort(key=lambda t: t[:5])

    out: list[Region] = []
    j = 0
    for i, b in enumerate(base):
        while j < len(placed) and placed[j][0] == i:
            out.append(placed[j][5])
            j += 1
        out.append(b)
    out.extend(p[5] for p in placed[j:])
    return out


# ---------------------------------------------------------------------------
# LayoutProcessor
# ---------------------------------------------------------------------------

class LayoutProcessor:
    """Post-process a PageLayout using battle-tested newspaper heuristics.

    Parameters
    ----------
    enabled:
        When False, :meth:`process` is a no-op (pass-through).
    confidence_thresh:
        Regions with ``confidence`` strictly below this value are dropped in
        the filter stage.  Matches the production pipeline default of 0.5.
    """

    def __init__(self, enabled: bool = True, confidence_thresh: float = 0.5) -> None:
        self.enabled = enabled
        self.confidence_thresh = confidence_thresh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, layout: PageLayout) -> PageLayout:
        """Run the full post-processing pipeline and return the (mutated) layout."""
        if not self.enabled:
            return layout

        if layout.ordered:
            return self._process_ordered(layout)

        regions = layout.regions
        regions = self._filter(regions)
        regions = self._rescue_low_confidence(regions, layout.regions)
        regions = self._deduplicate(regions)
        regions = self._fill_column_gaps(regions, layout.width, layout.height)
        regions = self._reading_order(regions)
        regions = self._merge_adjacent(regions, layout.image)
        regions = self._drop_empty_overlaps(regions, layout.lines_detected)
        layout.regions = regions
        return layout

    def _process_ordered(self, layout: PageLayout) -> PageLayout:
        """Post-process a layout whose detector supplied reading order.

        Applies the confidence filter/rescue as a membership test so surviving
        regions keep their original positions, then the empty-overlap drop.
        """
        kept = self._filter(layout.regions)
        kept = self._rescue_low_confidence(kept, layout.regions)
        keep_ids = {id(r) for r in kept}
        regions = [r for r in layout.regions if id(r) in keep_ids]
        layout.regions = self._drop_empty_overlaps(regions, layout.lines_detected)
        return layout

    # ------------------------------------------------------------------
    # Stage 1 – Filter
    # ------------------------------------------------------------------

    def _filter(self, regions: list[Region]) -> list[Region]:
        """Keep only regions at or above the confidence threshold."""
        kept = [r for r in regions if r.confidence >= self.confidence_thresh]
        return kept

    # ------------------------------------------------------------------
    # Stage 2 – Rescue low-confidence
    # ------------------------------------------------------------------

    def _rescue_low_confidence(
        self,
        accepted: list[Region],
        all_regions: list[Region],
        low_thresh: float = 0.15,
        max_overlap: float = 0.3,
    ) -> list[Region]:
        """Add low-confidence regions (between low_thresh and 0.5) that don't
        significantly overlap any already-accepted region.
        """
        candidates = [
            r
            for r in all_regions
            if low_thresh <= r.confidence < self.confidence_thresh
            and _box_area(_bbox_tuple(r)) >= 500
        ]

        rescued: list[Region] = []
        for cand in candidates:
            cand_bbox = _bbox_tuple(cand)
            cand_area = _box_area(cand_bbox)
            if cand_area == 0:
                continue
            total_overlap = sum(
                _intersection_area(cand_bbox, _bbox_tuple(acc)) for acc in accepted
            )
            if total_overlap / cand_area < max_overlap:
                rescued.append(cand)

        return accepted + rescued

    # ------------------------------------------------------------------
    # Stage 3 – Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(
        self,
        regions: list[Region],
        containment_thresh: float = 0.7,
        duplicate_thresh: float = 0.8,
    ) -> list[Region]:
        """Remove overlapping detections.

        Handles three cases:
        1. Near-duplicates with same label and high overlap -> keep higher confidence.
        2. Title overlapping non-title -> keep title.
        3. Contained box (small inside large, same label) -> remove smaller.
        """
        if len(regions) < 2:
            return regions

        title_labels = {"doc_title", "paragraph_title"}
        remove: set[int] = set()

        for i in range(len(regions)):
            if i in remove:
                continue
            for j in range(i + 1, len(regions)):
                if j in remove:
                    continue
                bi = _bbox_tuple(regions[i])
                bj = _bbox_tuple(regions[j])
                ai = _box_area(bi)
                aj = _box_area(bj)
                inter = _intersection_area(bi, bj)
                if inter == 0:
                    continue
                smaller_area = min(ai, aj)
                containment = inter / smaller_area if smaller_area > 0 else 0

                li = regions[i].label
                lj = regions[j].label

                # Case 2: title + text overlap -> keep title
                if containment > duplicate_thresh:
                    if li in title_labels and lj not in title_labels:
                        remove.add(j)
                        continue
                    if lj in title_labels and li not in title_labels:
                        remove.add(i)
                        break

                # Case 1: near-duplicates (same label) -> keep higher confidence
                if li == lj and containment > duplicate_thresh:
                    si = regions[i].confidence
                    sj = regions[j].confidence
                    if si >= sj:
                        remove.add(j)
                    else:
                        remove.add(i)
                        break
                    continue

                # Case 3: contained box (same label) -> remove smaller
                if li == lj and containment > containment_thresh:
                    if ai >= aj:
                        remove.add(j)
                    else:
                        remove.add(i)
                        break

                # Case 4: one region is mostly contained in another -> remove smaller
                if smaller_area > 0:
                    frac_i = inter / ai if ai > 0 else 0
                    frac_j = inter / aj if aj > 0 else 0
                    if frac_j > 0.5 and aj < ai:
                        remove.add(j)
                        continue
                    if frac_i > 0.5 and ai < aj:
                        remove.add(i)
                        break

                # Case 5: overlapping regions with very different sizes —
                # if one region is much larger and the overlap is significant
                # for the smaller region, drop the smaller one.
                larger_area = max(ai, aj)
                if smaller_area > 0 and larger_area / smaller_area > 3:
                    frac_small = inter / smaller_area
                    if frac_small > 0.3:
                        if ai < aj:
                            remove.add(i)
                            break
                        else:
                            remove.add(j)
                            continue

        return [r for i, r in enumerate(regions) if i not in remove]

    # ------------------------------------------------------------------
    # Stage 3b – Drop empty text regions that overlap content regions
    # ------------------------------------------------------------------

    def _drop_empty_overlaps(
        self, regions: list[Region], lines_detected: bool
    ) -> list[Region]:
        """Remove text-labeled regions the line detector found nothing in.

        A text region with no lines is a layout false positive — but only if a
        line detector actually ran.  When it didn't (``skip_lines=True``, or a
        region-only detector like PP-DocLayout) every region is line-less, and
        dropping them all would delete the page; those regions are instead left
        for the region-level OCR fallback in ``Pipeline.run``.
        """
        if not lines_detected:
            return regions
        return [r for r in regions if len(r.lines) > 0 or r.label not in _OCR_LABELS]

    # ------------------------------------------------------------------
    # Stage 4 – Fill column gaps
    # ------------------------------------------------------------------

    def _fill_column_gaps(
        self,
        regions: list[Region],
        img_w: int,
        img_h: int,
        min_gap_height: int = 80,
    ) -> list[Region]:
        """Find large vertical gaps within detected columns and add synthetic
        text regions so the reading-order and merge stages don't skip them.
        """
        if len(regions) < 3:
            return regions

        boxes = np.asarray([_bbox_tuple(r) for r in regions], dtype=int)
        col_centers, median_w = _find_columns(boxes)
        if not col_centers or len(col_centers) < 2:
            return regions

        new_regions = list(regions)
        # Use a 1x1 transparent placeholder image for synthetic regions.
        placeholder_img = Image.new("RGB", (1, 1))

        for center, cl, cr in col_centers:
            col_w = cr - cl
            col_boxes: list[tuple[int, int, int, int]] = []
            for r in regions:
                x1, y1, x2, y2 = _bbox_tuple(r)
                bw = x2 - x1
                if bw <= median_w * 1.3:
                    overlap = min(x2, cr) - max(x1, cl)
                    if col_w > 0 and overlap > col_w * 0.3:
                        col_boxes.append((y1, y2, x1, x2))
            if not col_boxes:
                continue
            col_boxes.sort()

            col_top = min(y1 for y1, y2, _, _ in col_boxes)
            col_bot = max(y2 for _, y2, _, _ in col_boxes)
            col_detected = sum(y2 - y1 for y1, y2, _, _ in col_boxes)
            col_span = col_bot - col_top
            col_coverage = col_detected / col_span if col_span > 0 else 1.0
            if col_coverage > 0.85:
                continue

            prev_bot = col_top
            for y1, y2, _, _ in col_boxes:
                gap = y1 - prev_bot
                if gap >= min_gap_height:
                    new_regions.append(
                        Region(
                            bbox=BBox(int(cl), int(prev_bot), int(cr), int(y1)),
                            image=placeholder_img,
                            label="text",
                            confidence=0.0,
                        )
                    )
                prev_bot = max(prev_bot, y2)

            content_bot = min(img_h - 20, col_bot + (col_bot - col_top) * 0.15)
            if content_bot - prev_bot >= min_gap_height:
                new_regions.append(
                    Region(
                        bbox=BBox(int(cl), int(prev_bot), int(cr), int(content_bot)),
                        image=placeholder_img,
                        label="text",
                        confidence=0.0,
                    )
                )

        return new_regions

    # ------------------------------------------------------------------
    # Stage 5 – Reading order
    # ------------------------------------------------------------------

    def _reading_order(self, regions: list[Region]) -> list[Region]:
        """Sort regions in newspaper column order (left column top-to-bottom,
        then right column, etc.) with full-width banners first.
        """
        if not regions:
            return regions

        bboxes = [_bbox_tuple(r) for r in regions]
        boxes = np.asarray(bboxes, dtype=int)
        n = len(boxes)
        if n <= 1:
            return regions

        col_centers, median_w = _find_columns(boxes)
        if not col_centers:
            # Fallback: simple top-to-bottom sort
            order = boxes[:, 1].argsort().tolist()
            return [regions[i] for i in order]

        num_cols = len(col_centers)

        assignments = []
        for i in range(n):
            x1, y1, x2, y2 = boxes[i]
            cols = _overlapping_cols(x1, x2, col_centers)
            assignments.append((i, cols, int(y1)))

        col_buckets: list[list[tuple[int, int]]] = [[] for _ in range(num_cols)]
        multi_col: list[tuple[int, int, int, int]] = []

        for idx, cols, y1 in assignments:
            if len(cols) == 1:
                col_buckets[cols[0]].append((y1, idx))
            else:
                multi_col.append((min(cols), max(cols), y1, idx))

        for bucket in col_buckets:
            bucket.sort()

        outputted: set[int] = set()
        result: list[int] = []

        # Full-width banners first, by Y position
        for first_c, last_c, y1, idx in sorted(multi_col, key=lambda x: x[2]):
            if last_c - first_c + 1 >= num_cols - 1:
                result.append(idx)
                outputted.add(idx)

        # Then column by column
        for c in range(num_cols):
            mc_for_col = [
                (y1, idx)
                for first_c, last_c, y1, idx in multi_col
                if first_c == c and idx not in outputted
            ]
            mc_for_col.sort()
            for _, idx in mc_for_col:
                result.append(idx)
                outputted.add(idx)
            for y1, idx in col_buckets[c]:
                result.append(idx)

        # Any remaining multi-column regions not yet output
        for first_c, last_c, y1, idx in sorted(multi_col, key=lambda x: x[2]):
            if idx not in outputted:
                result.append(idx)

        return [regions[i] for i in result]

    # ------------------------------------------------------------------
    # Stage 6 – Merge adjacent blocks
    # ------------------------------------------------------------------

    def _merge_adjacent(
        self,
        regions: list[Region],
        page_image: Image.Image,
        x_overlap_thresh: float = 0.5,
        y_gap_max: int = 30,
        max_height: int = 600,
    ) -> list[Region]:
        """Merge vertically adjacent, horizontally aligned text blocks.

        Title regions are never merged; they always terminate the current
        accumulator and are passed through as-is.
        """
        if not regions:
            return regions

        title_labels = {"doc_title", "paragraph_title"}
        merged: list[Region] = []
        current: Region | None = None

        for region in regions:
            label = region.label
            x1, y1, x2, y2 = _bbox_tuple(region)

            if label in title_labels:
                if current is not None:
                    merged.append(current)
                    current = None
                merged.append(region)
                continue

            if current is None:
                current = Region(
                    bbox=BBox(x1, y1, x2, y2),
                    image=region.image,
                    label=label,
                    lines=list(region.lines),
                    text=region.text,
                    confidence=region.confidence,
                    source=region.source,
                )
                continue

            cx1, cy1, cx2, cy2 = _bbox_tuple(current)
            cur_w = cx2 - cx1
            new_w = x2 - x1
            overlap = max(0, min(cx2, x2) - max(cx1, x1))
            overlap_ratio = overlap / min(cur_w, new_w) if min(cur_w, new_w) > 0 else 0
            y_gap = y1 - cy2
            merged_height = max(cy2, y2) - cy1

            if (
                overlap_ratio >= x_overlap_thresh
                and 0 <= y_gap <= y_gap_max
                and merged_height <= max_height
            ):
                # Expand current bounding box
                new_x0 = min(cx1, x1)
                new_y0 = cy1
                new_x1 = max(cx2, x2)
                new_y1 = max(cy2, y2)
                crop = page_image.crop((new_x0, new_y0, new_x1, new_y1))
                combined_text = (
                    (current.text + "\n" + region.text).strip()
                    if current.text or region.text
                    else ""
                )
                current = Region(
                    bbox=BBox(new_x0, new_y0, new_x1, new_y1),
                    image=crop,
                    label=current.label,
                    lines=current.lines + region.lines,
                    text=combined_text,
                    confidence=max(current.confidence, region.confidence),
                    # A block that absorbed any primary region counts as primary.
                    source=(current.source if current.source == region.source
                            else "primary"),
                )
            else:
                merged.append(current)
                current = Region(
                    bbox=BBox(x1, y1, x2, y2),
                    image=region.image,
                    label=label,
                    lines=list(region.lines),
                    text=region.text,
                    confidence=region.confidence,
                    source=region.source,
                )

        if current is not None:
            merged.append(current)

        return merged
