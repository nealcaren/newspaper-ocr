"""Layout post-processing for newspaper OCR pages.

Ported from the dangerouspress-ocr production pipeline (ocr_pipeline.py).
Adapts dict-based logic to use the Region / PageLayout data model.

Pipeline stages (in order):
  1. _filter          – drop regions below confidence threshold
  2. _rescue_low_confidence – re-admit low-conf regions that don't overlap accepted ones
  3. _deduplicate      – remove overlapping / contained duplicates
  4. _fill_column_gaps – add synthetic text regions for large vertical gaps in columns
  5. _reading_order    – sort regions in newspaper column order (top-to-bottom per column)
  6. _merge_adjacent   – merge vertically adjacent same-column text blocks
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from newspaper_ocr.models import BBox, PageLayout, Region

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

        regions = layout.regions
        regions = self._filter(regions)
        regions = self._rescue_low_confidence(regions, layout.regions)
        regions = self._deduplicate(regions)
        regions = self._fill_column_gaps(regions, layout.width, layout.height)
        regions = self._reading_order(regions)
        regions = self._merge_adjacent(regions, layout.image)
        layout.regions = regions
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

        return [r for i, r in enumerate(regions) if i not in remove]

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

        def overlapping_cols(x1: int, x2: int) -> list[int]:
            cols = []
            for c, (center, cl, cr) in enumerate(col_centers):
                overlap = min(x2, cr) - max(x1, cl)
                col_w = cr - cl
                if col_w > 0 and overlap > col_w * 0.2:
                    cols.append(c)
            return cols if cols else [
                min(range(num_cols), key=lambda c: abs(col_centers[c][0] - (x1 + x2) / 2))
            ]

        assignments = []
        for i in range(n):
            x1, y1, x2, y2 = boxes[i]
            cols = overlapping_cols(x1, x2)
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
                    label="text",
                    lines=list(region.lines),
                    text=region.text,
                    confidence=region.confidence,
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
                    label="text",
                    lines=current.lines + region.lines,
                    text=combined_text,
                    confidence=max(current.confidence, region.confidence),
                )
            else:
                merged.append(current)
                current = Region(
                    bbox=BBox(x1, y1, x2, y2),
                    image=region.image,
                    label="text",
                    lines=list(region.lines),
                    text=region.text,
                    confidence=region.confidence,
                )

        if current is not None:
            merged.append(current)

        return merged
