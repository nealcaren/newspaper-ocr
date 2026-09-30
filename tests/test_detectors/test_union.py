"""Tests for the hole-fill UnionDetector and order-preserving insertion."""

from __future__ import annotations

import numpy as np
from PIL import Image

from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.detectors.union import UnionDetector
from newspaper_ocr.layout_processor import LayoutProcessor, insert_in_order
from newspaper_ocr.models import BBox, PageLayout, Region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page(boxes_with_ink=(), w: int = 600, h: int = 400) -> Image.Image:
    """White page with solid black ink painted into each given box."""
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in boxes_with_ink:
        arr[y0:y1, x0:x1] = 0
    return Image.fromarray(arr)


def _r(x0, y0, x1, y1, name="", conf=0.9, label="text") -> Region:
    return Region(bbox=BBox(x0, y0, x1, y1), image=Image.new("RGB", (1, 1)),
                  label=label, confidence=conf, text=name)


class FixedDetector(Detector):
    """Detector returning fresh copies of a fixed region list."""

    def __init__(self, boxes, ordered=False, lines_detected=False):
        self.boxes = boxes
        self.ordered = ordered
        self.lines_detected = lines_detected

    def detect(self, image):
        regions = [_r(*b[:4], name=b[4] if len(b) > 4 else "") for b in self.boxes]
        return PageLayout(image=image, regions=regions, width=image.size[0],
                          height=image.size[1], ordered=self.ordered,
                          lines_detected=self.lines_detected)


def _names(regions):
    return [r.text for r in regions]


# Three 180px-wide columns, two stacked blocks each, in native reading order.
THREE_COLS = [
    (10, 10, 190, 190, "A1"), (10, 200, 190, 390, "A2"),
    (210, 10, 390, 190, "B1"), (210, 200, 390, 390, "B2"),
    (410, 10, 590, 190, "C1"), (410, 200, 590, 390, "C2"),
]


# ---------------------------------------------------------------------------
# Hole filtering
# ---------------------------------------------------------------------------

def test_candidate_covered_by_primary_is_not_a_hole():
    img = _page([(0, 0, 600, 400)])
    union = UnionDetector(FixedDetector([(0, 0, 300, 400, "P")]),
                          FixedDetector([(100, 100, 250, 200, "dup")]))
    layout = union.detect(img)
    assert _names(layout.regions) == ["P"]
    assert union.last_n_holes == 0


def test_overlap_threshold_boundary():
    # Candidate 100x100 at x=290..390; primary covers x<300, i.e. 10% of it.
    img = _page([(0, 0, 600, 400)])
    prim = FixedDetector([(0, 0, 300, 400, "P")])
    cand = FixedDetector([(290, 0, 390, 100, "H")])
    assert _names(UnionDetector(prim, cand).detect(img).regions) == ["P", "H"]
    # Tighten the threshold below 10% and it is rejected.
    assert _names(UnionDetector(prim, cand, overlap_max=0.05)
                  .detect(img).regions) == ["P"]


def test_blank_candidate_is_rejected_by_ink_filter():
    img = _page([(0, 0, 300, 400)])  # right half of the page is blank
    union = UnionDetector(FixedDetector([(0, 0, 300, 400, "P")]),
                          FixedDetector([(350, 50, 550, 350, "margin")]))
    assert _names(union.detect(img).regions) == ["P"]


def test_sparse_ink_threshold():
    # 200x100 candidate with a 10x10 ink blob = 0.5% ink.
    img = _page([(400, 100, 410, 110)])
    prim = FixedDetector([(0, 0, 100, 100, "P")])
    cand = FixedDetector([(400, 100, 600, 200, "H")])
    assert _names(UnionDetector(prim, cand).detect(img).regions) == ["P"]
    assert _names(UnionDetector(prim, cand, ink_min=0.004)
                  .detect(img).regions) == ["P", "H"]


def test_overlapping_candidates_yield_one_hole():
    img = _page([(300, 0, 600, 400)])
    prim = FixedDetector([(0, 0, 300, 400, "P")])
    sec = FixedDetector([])
    union = UnionDetector(prim, sec)
    holes = union.find_holes(
        img, prim.detect(img).regions,
        [_r(320, 20, 580, 380, "big", conf=0.9),
         _r(330, 30, 570, 370, "inner", conf=0.5)],
    )
    assert _names(holes) == ["big"]


def test_sources_tagged():
    img = _page([(0, 0, 600, 400)])
    layout = UnionDetector(FixedDetector([(0, 0, 300, 400, "P")]),
                           FixedDetector([(310, 0, 600, 400, "H")])).detect(img)
    assert [r.source for r in layout.regions] == ["primary", "hole"]


def test_no_holes_leaves_primary_unchanged():
    img = _page([(0, 0, 600, 400)])
    prim = FixedDetector(THREE_COLS, ordered=True)
    alone = prim.detect(img)
    union = UnionDetector(prim, FixedDetector([])).detect(img)
    assert [r.bbox for r in union.regions] == [r.bbox for r in alone.regions]
    assert union.ordered


def test_lines_detected_requires_both():
    img = _page()
    both = UnionDetector(FixedDetector([], lines_detected=True),
                         FixedDetector([], lines_detected=True)).detect(img)
    one = UnionDetector(FixedDetector([], lines_detected=True),
                        FixedDetector([], lines_detected=False)).detect(img)
    assert both.lines_detected and not one.lines_detected


# ---------------------------------------------------------------------------
# Hole placement
# ---------------------------------------------------------------------------

def test_hole_inserted_by_column_and_y_when_ordered():
    # B has a gap at y=190..200 widened: drop B2, put a hole where it was.
    prim_boxes = [b for b in THREE_COLS if b[4] != "B2"]
    img = _page([(0, 0, 600, 400)])
    layout = UnionDetector(FixedDetector(prim_boxes, ordered=True),
                           FixedDetector([(210, 200, 390, 390, "hole")])).detect(img)
    assert _names(layout.regions) == ["A1", "A2", "B1", "hole", "C1", "C2"]


def test_hole_at_top_of_column_goes_before_it():
    prim_boxes = [b for b in THREE_COLS if b[4] != "C1"]
    img = _page([(0, 0, 600, 400)])
    layout = UnionDetector(FixedDetector(prim_boxes, ordered=True),
                           FixedDetector([(410, 10, 590, 190, "hole")])).detect(img)
    assert _names(layout.regions) == ["A1", "A2", "B1", "B2", "hole", "C2"]


def test_insertion_keeps_non_geometric_base_order():
    # The base order is not a geometric sort (C before B); insertion must
    # not re-sort it.
    base = [_r(*THREE_COLS[i][:4], name=THREE_COLS[i][4]) for i in (0, 1, 4, 5, 2, 3)]
    out = insert_in_order(base, [])
    assert _names(out) == ["A1", "A2", "C1", "C2", "B1", "B2"]


def test_multiple_holes_same_slot_ordered_by_column_then_y():
    base = [_r(*b[:4], name=b[4]) for b in THREE_COLS if b[4].startswith("A")]
    extras = [_r(410, 200, 590, 390, name="C2"), _r(210, 10, 390, 190, name="B1"),
              _r(410, 10, 590, 190, name="C1")]
    out = insert_in_order(base, extras)
    assert _names(out) == ["A1", "A2", "B1", "C1", "C2"]


def test_unordered_primary_appends_holes_for_layout_sort():
    img = _page([(0, 0, 600, 400)])
    layout = UnionDetector(FixedDetector([(210, 10, 390, 390, "B")]),
                           FixedDetector([(10, 10, 190, 390, "A")])).detect(img)
    assert not layout.ordered
    assert _names(layout.regions) == ["B", "A"]


# ---------------------------------------------------------------------------
# LayoutProcessor respects native order
# ---------------------------------------------------------------------------

def test_layout_processor_keeps_ordered_layout():
    regions = [_r(*THREE_COLS[i][:4], name=THREE_COLS[i][4]) for i in (4, 0, 2, 5, 1, 3)]
    layout = PageLayout(image=_page(), regions=regions, width=600, height=400,
                        ordered=True)
    out = LayoutProcessor().process(layout)
    assert _names(out.regions) == ["C1", "A1", "B1", "C2", "A2", "B2"]


def test_layout_processor_ordered_still_filters_low_confidence_in_place():
    regions = [_r(10, 10, 190, 190, "keep1"),
               _r(10, 200, 190, 390, "drop", conf=0.05),
               _r(210, 10, 390, 190, "keep2")]
    layout = PageLayout(image=_page(), regions=regions, width=600, height=400,
                        ordered=True)
    assert _names(LayoutProcessor().process(layout).regions) == ["keep1", "keep2"]


def test_merge_keeps_source():
    a = _r(10, 10, 190, 100, "a"); a.source = "primary"
    b = _r(10, 110, 190, 200, "b"); b.source = "hole"
    c = _r(410, 10, 590, 100, "c"); c.source = "hole"
    out = LayoutProcessor()._merge_adjacent([a, b, c], _page())
    assert [r.source for r in out] == ["primary", "hole"]
