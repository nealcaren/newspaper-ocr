"""Tests for LayoutProcessor."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from newspaper_ocr.layout_processor import LayoutProcessor
from newspaper_ocr.models import BBox, PageLayout, Region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img(w: int = 10, h: int = 10) -> Image.Image:
    """Return a blank RGB image of the given size."""
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def _region(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    label: str = "text",
    confidence: float = 0.9,
    text: str = "",
) -> Region:
    return Region(
        bbox=BBox(x0, y0, x1, y1),
        image=_img(x1 - x0, y1 - y0),
        label=label,
        confidence=confidence,
        text=text,
    )


def _layout(regions: list[Region], w: int = 1000, h: int = 1500) -> PageLayout:
    return PageLayout(image=_img(w, h), regions=regions, width=w, height=h)


# ---------------------------------------------------------------------------
# Enabled / disabled
# ---------------------------------------------------------------------------

def test_disabled_is_passthrough():
    """When enabled=False, process() returns the layout unchanged."""
    lp = LayoutProcessor(enabled=False)
    r = _region(0, 0, 100, 100, confidence=0.0)
    layout = _layout([r])
    result = lp.process(layout)
    assert result.regions == [r]


def test_enabled_by_default():
    lp = LayoutProcessor()
    assert lp.enabled is True


# ---------------------------------------------------------------------------
# Stage 1: _filter
# ---------------------------------------------------------------------------

def test_filter_removes_low_confidence():
    lp = LayoutProcessor(confidence_thresh=0.5)
    regions = [
        _region(0, 0, 200, 100, confidence=0.8),
        _region(0, 100, 200, 200, confidence=0.3),  # below threshold
        _region(0, 200, 200, 300, confidence=0.5),  # exactly at threshold -> kept
    ]
    kept = lp._filter(regions)
    assert len(kept) == 2
    assert all(r.confidence >= 0.5 for r in kept)


def test_filter_keeps_all_above_threshold():
    lp = LayoutProcessor(confidence_thresh=0.5)
    regions = [_region(0, i * 100, 200, (i + 1) * 100, confidence=0.9) for i in range(5)]
    kept = lp._filter(regions)
    assert len(kept) == 5


def test_filter_drops_all_below_threshold():
    lp = LayoutProcessor(confidence_thresh=0.9)
    regions = [_region(0, 0, 200, 100, confidence=0.1)]
    kept = lp._filter(regions)
    assert kept == []


# ---------------------------------------------------------------------------
# Stage 2: _rescue_low_confidence
# ---------------------------------------------------------------------------

def test_rescue_adds_non_overlapping_low_confidence():
    lp = LayoutProcessor(confidence_thresh=0.5)
    accepted = [_region(0, 0, 200, 100, confidence=0.9)]
    # This low-conf region is far away from accepted ones
    low_conf = [_region(300, 300, 500, 400, confidence=0.3)]
    rescued = lp._rescue_low_confidence(accepted, accepted + low_conf)
    assert len(rescued) == 2


def test_rescue_skips_heavily_overlapping():
    lp = LayoutProcessor(confidence_thresh=0.5)
    accepted = [_region(0, 0, 200, 200, confidence=0.9)]
    # Entirely inside the accepted region
    low_conf = _region(10, 10, 190, 190, confidence=0.3)
    result = lp._rescue_low_confidence(accepted, accepted + [low_conf])
    assert len(result) == 1  # low_conf not added


def test_rescue_ignores_too_small_candidates():
    lp = LayoutProcessor(confidence_thresh=0.5)
    accepted: list[Region] = []
    tiny = _region(0, 0, 10, 5, confidence=0.3)  # area=50, below 500 threshold
    result = lp._rescue_low_confidence(accepted, [tiny])
    assert result == []


# ---------------------------------------------------------------------------
# Stage 3: _deduplicate
# ---------------------------------------------------------------------------

def test_deduplicate_keeps_higher_confidence():
    lp = LayoutProcessor()
    # Two heavily overlapping same-label regions
    high = _region(0, 0, 200, 200, confidence=0.9)
    low = _region(5, 5, 195, 195, confidence=0.4)
    result = lp._deduplicate([high, low])
    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_deduplicate_keeps_lower_confidence_if_higher():
    lp = LayoutProcessor()
    low = _region(0, 0, 200, 200, confidence=0.4)
    high = _region(5, 5, 195, 195, confidence=0.9)
    result = lp._deduplicate([low, high])
    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_deduplicate_title_beats_text():
    lp = LayoutProcessor()
    title = _region(0, 0, 200, 80, label="paragraph_title", confidence=0.8)
    text = _region(5, 5, 195, 75, label="text", confidence=0.95)
    result = lp._deduplicate([title, text])
    assert len(result) == 1
    assert result[0].label == "paragraph_title"


def test_deduplicate_no_overlap_keeps_both():
    lp = LayoutProcessor()
    r1 = _region(0, 0, 100, 100)
    r2 = _region(200, 200, 300, 300)
    result = lp._deduplicate([r1, r2])
    assert len(result) == 2


def test_deduplicate_single_region():
    lp = LayoutProcessor()
    r = _region(0, 0, 100, 100)
    assert lp._deduplicate([r]) == [r]


# ---------------------------------------------------------------------------
# Stage 5: _reading_order (two-column layout)
# ---------------------------------------------------------------------------

def _two_column_layout(
    n_per_col: int = 4,
    col_w: int = 200,
    row_h: int = 100,
    gap: int = 50,
) -> tuple[PageLayout, list[Region]]:
    """Build a simple two-column layout with n_per_col rows each.

    Left column:  x = 0..col_w
    Right column: x = col_w+gap .. 2*col_w+gap
    Regions are created interleaved in list order to ensure reading order
    is not already correct.
    """
    left_regions = [
        _region(0, i * row_h, col_w, (i + 1) * row_h, text=f"L{i}")
        for i in range(n_per_col)
    ]
    right_regions = [
        _region(col_w + gap, i * row_h, 2 * col_w + gap, (i + 1) * row_h, text=f"R{i}")
        for i in range(n_per_col)
    ]
    # Interleave: R0, L0, R1, L1, ...  (wrong order intentionally)
    interleaved = []
    for l, r in zip(left_regions, right_regions):
        interleaved.extend([r, l])
    img_w = 2 * col_w + gap
    img_h = n_per_col * row_h
    layout = _layout(interleaved, w=img_w, h=img_h)
    return layout, left_regions + right_regions


def test_reading_order_left_column_first():
    """Left-column regions should all appear before right-column regions."""
    lp = LayoutProcessor()
    layout, _ = _two_column_layout(n_per_col=4)
    ordered = lp._reading_order(layout.regions)
    texts = [r.text for r in ordered]
    # All L* should come before any R*
    left_indices = [i for i, t in enumerate(texts) if t.startswith("L")]
    right_indices = [i for i, t in enumerate(texts) if t.startswith("R")]
    assert left_indices, "No left-column regions found"
    assert right_indices, "No right-column regions found"
    assert max(left_indices) < min(right_indices), (
        f"Left column not fully before right column: {texts}"
    )


def test_reading_order_top_to_bottom_within_column():
    """Within each column, regions should be sorted top-to-bottom."""
    lp = LayoutProcessor()
    layout, _ = _two_column_layout(n_per_col=4)
    ordered = lp._reading_order(layout.regions)
    left_ordered = [r for r in ordered if r.text.startswith("L")]
    right_ordered = [r for r in ordered if r.text.startswith("R")]
    for col_regions in (left_ordered, right_ordered):
        y_tops = [r.bbox.y0 for r in col_regions]
        assert y_tops == sorted(y_tops), f"Not top-to-bottom: {y_tops}"


def test_reading_order_single_column_fallback():
    """Single-column page should be sorted top-to-bottom."""
    lp = LayoutProcessor()
    # All same width -> single column detected
    regions = [_region(0, i * 100, 800, (i + 1) * 100, text=str(i)) for i in [3, 1, 0, 2]]
    layout = _layout(regions, w=800, h=400)
    ordered = lp._reading_order(layout.regions)
    y_tops = [r.bbox.y0 for r in ordered]
    assert y_tops == sorted(y_tops)


def test_reading_order_empty():
    lp = LayoutProcessor()
    assert lp._reading_order([]) == []


# ---------------------------------------------------------------------------
# Stage 6: _merge_adjacent
# ---------------------------------------------------------------------------

def test_merge_adjacent_combines_close_blocks():
    lp = LayoutProcessor()
    page = _img(200, 500)
    layout = _layout([], w=200, h=500)
    # Two vertically adjacent blocks with the same width -> should merge
    r1 = _region(0, 0, 200, 100, text="Hello")
    r2 = _region(0, 110, 200, 200, text="world")  # gap = 10 px
    merged = lp._merge_adjacent([r1, r2], page)
    assert len(merged) == 1
    assert "Hello" in merged[0].text
    assert "world" in merged[0].text


def test_merge_adjacent_does_not_merge_title():
    lp = LayoutProcessor()
    page = _img(200, 400)
    title = _region(0, 0, 200, 50, label="paragraph_title", text="Title")
    body = _region(0, 60, 200, 150, label="text", text="Body")
    merged = lp._merge_adjacent([title, body], page)
    # Title is a standalone block; body may start a new accumulator
    labels = [r.label for r in merged]
    assert "paragraph_title" in labels


def test_merge_adjacent_respects_y_gap():
    lp = LayoutProcessor()
    page = _img(200, 600)
    r1 = _region(0, 0, 200, 100, text="A")
    r2 = _region(0, 200, 200, 300, text="B")  # gap = 100 px > y_gap_max=30
    merged = lp._merge_adjacent([r1, r2], page)
    assert len(merged) == 2


def test_merge_adjacent_respects_max_height():
    lp = LayoutProcessor()
    page = _img(200, 1500)
    # Stack 10 blocks, each 80 px tall with 5 px gap; merged height would be ~850 px > 600
    regions = [_region(0, i * 85, 200, i * 85 + 80, text=f"block{i}") for i in range(10)]
    merged = lp._merge_adjacent(regions, page)
    for r in merged:
        assert r.bbox.height <= 600 + 80  # allow one block over due to boundary


def test_merge_adjacent_empty():
    lp = LayoutProcessor()
    page = _img(200, 200)
    assert lp._merge_adjacent([], page) == []


def test_merge_adjacent_single_region():
    lp = LayoutProcessor()
    page = _img(200, 200)
    r = _region(0, 0, 200, 100, text="only")
    result = lp._merge_adjacent([r], page)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# End-to-end process()
# ---------------------------------------------------------------------------

def test_process_filters_low_confidence_regions():
    """Regions below low_thresh (0.15) are dropped entirely by filter and
    cannot be rescued (rescue only considers confidence >= 0.15 and < thresh).
    """
    lp = LayoutProcessor(confidence_thresh=0.5)
    regions = [
        _region(0, 0, 500, 200, confidence=0.9, text="kept"),
        # confidence below the rescue floor of 0.15 -> dropped permanently
        _region(0, 300, 500, 500, confidence=0.05, text="removed"),
    ]
    layout = _layout(regions)
    result = lp.process(layout)
    texts = [r.text for r in result.regions]
    assert "kept" in " ".join(texts)
    assert "removed" not in " ".join(texts)


def test_process_returns_pagelayout():
    lp = LayoutProcessor()
    layout = _layout([_region(0, 0, 200, 100)])
    result = lp.process(layout)
    assert isinstance(result, PageLayout)


def test_process_empty_layout():
    lp = LayoutProcessor()
    layout = _layout([])
    result = lp.process(layout)
    assert result.regions == []
