"""Tests for post-recognition region repair."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from newspaper_ocr.models import TIMEOUT_TEXT, BBox, Line, PageLayout, Region
from newspaper_ocr.region_repair import (
    DuplicatePage,
    RegionRepair,
    find_duplicate_pages,
    page_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img(w: int = 10, h: int = 10) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def _region(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    text: str = "",
    label: str = "text",
    status: str = "ok",
    confidence: float = 0.9,
    id: str = "",
) -> Region:
    return Region(
        bbox=BBox(x0, y0, x1, y1),
        image=_img(max(1, x1 - x0), max(1, y1 - y0)),
        label=label,
        text=text,
        status=status,
        confidence=confidence,
        id=id,
    )


def _layout(regions: list[Region], w: int = 1000, h: int = 2000) -> PageLayout:
    return PageLayout(image=_img(w, h), regions=regions, width=w, height=h)


class _Recognizer:
    """Re-OCR double: returns canned text and records every crop it was given."""

    def __init__(self, text: str = "recovered text", status: str = "ok"):
        self.text = text
        self.status = status
        self.crops: list[tuple[int, int]] = []

    def __call__(self, crop: Image.Image) -> tuple[str, str]:
        self.crops.append(crop.size)
        return self.text, self.status


# A body paragraph long enough to clear the substring-dedup floor.
PARA_A = "the convention opened on tuesday with delegates from every division"
PARA_B = "resolutions were adopted without a dissenting vote before adjournment"


# ---------------------------------------------------------------------------
# Enabled / non-destructive contract
# ---------------------------------------------------------------------------

def test_disabled_is_passthrough():
    r = _region(0, 0, 100, 100, text=PARA_A)
    layout = _layout([r, _region(0, 0, 100, 100, text=PARA_A)])
    out = RegionRepair(enabled=False).repair(layout)
    assert len(out.regions) == 2
    assert out.raw_regions is None


def test_repair_snapshots_raw_regions():
    """The recognized regions survive repair untouched, as raw_regions."""
    keep = _region(0, 0, 100, 100, text=PARA_A)
    dup = _region(0, 0, 100, 100, text=PARA_A)
    layout = _layout([keep, dup])

    RegionRepair().repair(layout)

    assert len(layout.regions) == 1
    assert layout.raw_regions is not None
    assert [r.text for r in layout.raw_regions] == [PARA_A, PARA_A]


def test_repair_is_idempotent_and_recomputes_from_raw():
    """A second run gives the same answer, and re-running with looser thresholds
    resurrects regions the first run dropped — repair never eats its own input."""
    layout = _layout(
        [
            _region(0, 0, 100, 100, text=PARA_A),
            _region(0, 0, 100, 100, text=PARA_A),
        ]
    )

    RegionRepair().repair(layout)
    assert len(layout.regions) == 1

    RegionRepair().repair(layout)
    assert len(layout.regions) == 1

    # duplicate_iou above 1.0 can never fire: both regions come back.
    RegionRepair(duplicate_iou=1.5).repair(layout)
    assert len(layout.regions) == 2


def test_repaired_regions_are_copies():
    """Mutating a repaired region cannot reach back into the raw layer."""
    layout = _layout([_region(0, 0, 100, 100, text=PARA_A)])
    RegionRepair().repair(layout)

    layout.regions[0].text = "clobbered"
    assert layout.raw_regions[0].text == PARA_A


# ---------------------------------------------------------------------------
# Pass 1 – lossless text dedup
# ---------------------------------------------------------------------------

def test_identical_text_same_place_keeps_better_status():
    good = _region(0, 0, 100, 100, text=PARA_A, status="ok")
    bad = _region(2, 2, 102, 102, text=PARA_A, status="repetition")
    kept = RegionRepair().dedupe_text([bad, good])
    assert kept == [good]


def test_identical_text_far_apart_is_kept():
    """A standing head repeated elsewhere on the page is not a duplicate detection."""
    a = _region(0, 0, 100, 40, text=PARA_A)
    b = _region(500, 900, 600, 940, text=PARA_A)
    assert len(RegionRepair().dedupe_text([a, b])) == 2


def test_substring_inside_overlapping_region_is_dropped():
    container = _region(0, 0, 200, 600, text=f"{PARA_A} {PARA_B}")
    inner = _region(0, 0, 200, 300, text=PARA_A)
    assert RegionRepair().dedupe_text([container, inner]) == [container]


def test_substring_elsewhere_on_the_page_is_kept():
    container = _region(0, 0, 200, 600, text=f"{PARA_A} {PARA_B}")
    elsewhere = _region(700, 1200, 900, 1400, text=PARA_A)
    assert len(RegionRepair().dedupe_text([container, elsewhere])) == 2


def test_short_substring_is_kept():
    """Short strings turn up inside longer ones by coincidence."""
    container = _region(0, 0, 200, 600, text=f"{PARA_A} the end")
    short = _region(0, 0, 200, 100, text="the end")
    assert len(RegionRepair().dedupe_text([container, short])) == 2


def test_fuzzy_overlap_is_never_dropped():
    """Two reads of one passage each carry tokens the other lacks — keep both."""
    column = _region(0, 0, 200, 600, text="the convention opened on tuesday with delegates")
    paragraph = _region(0, 0, 200, 300, text="the convention opencd on tuesday witn delegates")
    assert len(RegionRepair().dedupe_text([column, paragraph])) == 2


def test_timeout_placeholders_are_not_duplicates_of_each_other():
    a = _region(0, 0, 100, 100, text=TIMEOUT_TEXT, status="timeout")
    b = _region(2, 2, 102, 102, text=TIMEOUT_TEXT, status="timeout")
    assert len(RegionRepair().dedupe_text([a, b])) == 2


def test_dedup_ignores_case_and_whitespace():
    a = _region(0, 0, 100, 100, text=PARA_A, status="ok")
    b = _region(0, 0, 100, 100, text=PARA_A.upper().replace(" ", "\n"), status="error")
    assert RegionRepair().dedupe_text([a, b]) == [a]


# ---------------------------------------------------------------------------
# Pass 2 – container split
# ---------------------------------------------------------------------------

def _column_page(container_text: str = "column read") -> tuple[PageLayout, Region]:
    """A tall column region with two paragraph reads covering its top 70%."""
    container = _region(100, 0, 300, 1000, text=container_text, id="r0")
    inner_a = _region(100, 0, 300, 350, text=PARA_A, id="r1")
    inner_b = _region(100, 360, 300, 700, text=PARA_B, id="r2")
    return _layout([container, inner_a, inner_b]), container


def test_container_with_gap_is_split_and_gap_reocrd():
    layout, container = _column_page()
    rec = _Recognizer("text nobody else read")

    RegionRepair().repair(layout, rec)

    assert container not in layout.regions
    strips = [r for r in layout.regions if r.text == "text nobody else read"]
    assert len(strips) == 1
    # The uncovered strip is the container's tail, at its full width.
    assert strips[0].bbox.to_tuple() == (100, 700, 300, 1000)
    assert rec.crops == [(200, 300)]
    # Derived from the container it came from, and unique on the page.
    assert strips[0].id == "r0s0"
    assert len({r.id for r in layout.regions}) == len(layout.regions)


def test_fully_covered_container_is_dropped_without_reocr():
    container = _region(100, 0, 300, 700, text="column read", id="r0")
    inner_a = _region(100, 0, 300, 350, text=PARA_A, id="r1")
    inner_b = _region(100, 350, 300, 700, text=PARA_B, id="r2")
    layout = _layout([container, inner_a, inner_b])
    rec = _Recognizer()

    RegionRepair().repair(layout, rec)

    assert [r.id for r in layout.regions] == ["r1", "r2"]
    assert rec.crops == []


def test_blank_strip_is_not_kept_as_an_empty_region():
    layout, container = _column_page()
    rec = _Recognizer("   ")

    RegionRepair().repair(layout, rec)

    assert [r.id for r in layout.regions] == ["r1", "r2"]


def test_container_survives_when_strip_reocr_fails():
    """Dropping a container whose gap could not be re-read is the data loss
    this stage exists to prevent."""
    layout, container = _column_page()
    rec = _Recognizer(TIMEOUT_TEXT, status="timeout")

    RegionRepair().repair(layout, rec)

    assert container.id in [r.id for r in layout.regions]


def test_container_survives_without_a_recognizer():
    layout, container = _column_page()
    RegionRepair().repair(layout, recognize=None)
    assert container.id in [r.id for r in layout.regions]


def test_lightly_covered_region_is_not_a_container():
    container = _region(100, 0, 300, 1000, text="column read", id="r0")
    inner_a = _region(100, 0, 300, 150, text=PARA_A, id="r1")
    inner_b = _region(100, 160, 300, 300, text=PARA_B, id="r2")
    layout = _layout([container, inner_a, inner_b])

    RegionRepair().repair(layout, _Recognizer())

    assert "r0" in [r.id for r in layout.regions]


def test_neighbouring_column_does_not_count_as_inner():
    """Coverage is per column: regions in the next column over prove nothing."""
    container = _region(100, 0, 300, 1000, text="column read", id="r0")
    other_col_a = _region(400, 0, 600, 500, text=PARA_A, id="r1")
    other_col_b = _region(400, 500, 600, 1000, text=PARA_B, id="r2")
    layout = _layout([container, other_col_a, other_col_b])

    RegionRepair().repair(layout, _Recognizer())

    assert "r0" in [r.id for r in layout.regions]


def test_empty_inner_regions_do_not_cover_a_container():
    """Empty detections cover geometry, not words — a container they 'cover'
    is the only region holding that text."""
    container = _region(100, 0, 300, 700, text="column read", id="r0")
    empty_a = _region(100, 0, 300, 350, text="", id="r1")
    empty_b = _region(100, 350, 300, 700, text="", id="r2")
    layout = _layout([container, empty_a, empty_b])

    RegionRepair().repair(layout, _Recognizer())

    assert "r0" in [r.id for r in layout.regions]


def test_single_inner_region_is_not_enough():
    container = _region(100, 0, 300, 700, text="column read", id="r0")
    inner = _region(100, 0, 300, 650, text=PARA_A, id="r1")
    layout = _layout([container, inner])

    RegionRepair().repair(layout, _Recognizer())

    assert "r0" in [r.id for r in layout.regions]


def test_sliver_gaps_are_not_reocrd():
    """A few pixels between paragraphs is a gutter, not missed text."""
    container = _region(100, 0, 300, 700, text="column read", id="r0")
    inner_a = _region(100, 0, 300, 340, text=PARA_A, id="r1")
    inner_b = _region(100, 350, 300, 700, text=PARA_B, id="r2")
    layout = _layout([container, inner_a, inner_b])
    rec = _Recognizer()

    RegionRepair().repair(layout, rec)

    assert rec.crops == []
    assert [r.id for r in layout.regions] == ["r1", "r2"]


def test_non_text_containers_are_left_alone():
    container = _region(100, 0, 300, 1000, text="ad read", label="image", id="r0")
    inner_a = _region(100, 0, 300, 350, text=PARA_A, id="r1")
    inner_b = _region(100, 360, 300, 700, text=PARA_B, id="r2")
    layout = _layout([container, inner_a, inner_b])

    RegionRepair().repair(layout, _Recognizer())

    assert "r0" in [r.id for r in layout.regions]


# ---------------------------------------------------------------------------
# Pass 3 – fragmented-ad merge
# ---------------------------------------------------------------------------

def _ad_page() -> PageLayout:
    """Two overlapping shards of one display ad."""
    return _layout(
        [
            _region(100, 100, 400, 400, text="BUY", label="image", id="r0"),
            _region(150, 150, 450, 450, text="NOW", label="image", id="r1"),
        ]
    )


def test_overlapping_fragments_merge_into_one_reocrd_region():
    layout = _ad_page()
    rec = _Recognizer("BUY NOW AT THE UNIVERSAL STORE")

    RegionRepair().repair(layout, rec)

    assert len(layout.regions) == 1
    merged = layout.regions[0]
    assert merged.text == "BUY NOW AT THE UNIVERSAL STORE"
    assert merged.bbox.to_tuple() == (100, 100, 450, 450)
    assert merged.id == "r0m"
    assert len(rec.crops) == 1


def test_merge_is_abandoned_when_the_union_would_swallow_a_neighbour():
    """Re-OCRing a union that contains a non-member duplicates its text."""
    layout = _ad_page()
    bystander = _region(160, 160, 260, 260, text=PARA_A, label="text", id="r2")
    layout.regions.append(bystander)
    rec = _Recognizer("BUY NOW")

    RegionRepair().repair(layout, rec)

    assert [r.id for r in layout.regions] == ["r0", "r1", "r2"]
    assert rec.crops == []


def test_merge_is_abandoned_when_the_union_covers_most_of_the_page():
    layout = _layout(
        [
            _region(0, 0, 900, 1200, text="a", label="image", id="r0"),
            _region(100, 100, 950, 1900, text="b", label="image", id="r1"),
        ]
    )
    rec = _Recognizer("whole page")

    RegionRepair().repair(layout, rec)

    assert [r.id for r in layout.regions] == ["r0", "r1"]
    assert rec.crops == []


def test_barely_overlapping_fragments_are_not_merged():
    layout = _layout(
        [
            _region(100, 100, 400, 400, text="a", label="image", id="r0"),
            _region(380, 380, 680, 680, text="b", label="image", id="r1"),
        ]
    )
    rec = _Recognizer()

    RegionRepair().repair(layout, rec)

    assert [r.id for r in layout.regions] == ["r0", "r1"]
    assert rec.crops == []


def test_merge_is_abandoned_when_reocr_returns_nothing():
    layout = _ad_page()
    RegionRepair().repair(layout, _Recognizer("", status="ok"))
    assert [r.id for r in layout.regions] == ["r0", "r1"]


def test_merge_labels_restrict_the_pass():
    layout = _ad_page()
    RegionRepair(merge_labels={"figure"}).repair(layout, _Recognizer("BUY NOW"))
    assert [r.id for r in layout.regions] == ["r0", "r1"]


def test_fragments_merge_without_a_recognizer_is_a_noop():
    layout = _ad_page()
    RegionRepair().repair(layout, recognize=None)
    assert [r.id for r in layout.regions] == ["r0", "r1"]


def test_merge_keeps_reading_order_position():
    layout = _layout(
        [
            _region(0, 0, 90, 90, text=PARA_A, label="text", id="r0"),
            _region(100, 100, 400, 400, text="BUY", label="image", id="r1"),
            _region(150, 150, 450, 450, text="NOW", label="image", id="r2"),
            _region(600, 600, 700, 700, text=PARA_B, label="text", id="r3"),
        ]
    )
    RegionRepair().repair(layout, _Recognizer("BUY NOW"))
    assert [r.id for r in layout.regions] == ["r0", "r1m", "r3"]


# ---------------------------------------------------------------------------
# Pass 4 – duplicate page scans
# ---------------------------------------------------------------------------

_PAGE_ONE = (
    "THE NEGRO WORLD. Convention of delegates opened on Tuesday morning at "
    "Liberty Hall with more than three hundred representatives present from "
    "divisions in every state and from the islands of the Caribbean. "
) * 3
_PAGE_TWO = (
    "SHIPPING NOTICES. The steamship line announces sailings for the month of "
    "June, calling at Kingston, Colon and Port Limon, with freight rates "
    "quoted on application at the office on West One Hundred and Thirty. "
) * 3


def _rescan(text: str) -> str:
    """The same page shot again: same words, OCR'd slightly differently."""
    return text.replace("e", "c", 12).replace("Tuesday", "Tucsday")


def test_duplicate_scan_is_found():
    found = find_duplicate_pages([_PAGE_ONE, _PAGE_TWO, _rescan(_PAGE_ONE)])
    assert found == [DuplicatePage(index=2, duplicate_of=0, similarity=pytest.approx(found[0].similarity))]
    assert found[0].similarity >= 0.35


def test_distinct_pages_are_not_duplicates():
    assert find_duplicate_pages([_PAGE_ONE, _PAGE_TWO]) == []


def test_each_rescan_points_at_the_first_copy():
    found = find_duplicate_pages([_PAGE_ONE, _rescan(_PAGE_ONE), _rescan(_PAGE_ONE)])
    assert [(d.index, d.duplicate_of) for d in found] == [(1, 0), (2, 0)]


def test_short_pages_are_skipped():
    """Two nearly-blank pages match perfectly and mean nothing."""
    assert find_duplicate_pages(["", ""]) == []
    assert find_duplicate_pages(["[OCR timeout]", "[OCR timeout]"]) == []


def test_page_layouts_are_accepted_directly():
    a = _layout([_region(0, 0, 100, 100, text=_PAGE_ONE)])
    b = _layout([_region(0, 0, 100, 100, text=_rescan(_PAGE_ONE))])
    assert [d.index for d in find_duplicate_pages([a, b])] == [1]


def test_page_similarity_separates_rescans_from_distinct_pages():
    assert page_similarity(_PAGE_ONE, _rescan(_PAGE_ONE)) > 0.35
    assert page_similarity(_PAGE_ONE, _PAGE_TWO) < 0.35


def test_prefilter_does_not_hide_a_real_duplicate():
    """The quick_ratio pre-filters are upper bounds, so they may only skip
    comparisons ratio() would also have rejected."""
    pages = [_PAGE_ONE, _PAGE_TWO, _rescan(_PAGE_ONE), _rescan(_PAGE_TWO)]
    found = {(d.index, d.duplicate_of) for d in find_duplicate_pages(pages)}
    assert found == {(2, 0), (3, 1)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_counts_what_changed():
    layout, _ = _column_page()
    repair = RegionRepair()
    repair.repair(layout, _Recognizer("tail of the column"))

    report = repair.last_report
    assert report.containers_split == 1
    assert report.strips_added == 1
    assert report.regions_before == 3
    assert report.regions_after == 3
    assert report.changed is True


def test_report_is_quiet_when_nothing_changes():
    repair = RegionRepair()
    repair.repair(_layout([_region(0, 0, 100, 100, text=PARA_A, id="r0")]))
    assert repair.last_report.changed is False


# ---------------------------------------------------------------------------
# Regression: the lines a repaired region carries
# ---------------------------------------------------------------------------

def test_dedup_keeps_line_level_detail():
    keep = _region(0, 0, 100, 100, text=PARA_A)
    keep.lines = [Line(bbox=BBox(0, 0, 100, 20), image=_img(100, 20), text=PARA_A)]
    layout = _layout([keep, _region(0, 0, 100, 100, text=PARA_A, status="error")])

    RegionRepair().repair(layout)

    assert len(layout.regions) == 1
    assert [line.text for line in layout.regions[0].lines] == [PARA_A]
