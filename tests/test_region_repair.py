"""Tests for the post-recognition RegionRepair stage."""

from __future__ import annotations

from PIL import Image

from newspaper_ocr.models import BBox, PageLayout, Region
from newspaper_ocr.recognizers.base import RegionRecognizer
from newspaper_ocr.region_repair import RegionRepair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(x0, y0, x1, y1, text="", status="ok", label="text") -> Region:
    return Region(bbox=BBox(x0, y0, x1, y1), image=Image.new("RGB", (1, 1)),
                  label=label, text=text, status=status)


def _layout(regions, w=1000, h=1000, ordered=False, lines_detected=False):
    return PageLayout(image=Image.new("RGB", (w, h), "white"), regions=regions,
                      width=w, height=h, ordered=ordered,
                      lines_detected=lines_detected)


class FakeRecognizer(RegionRecognizer):
    """Returns ``texts[bbox tuple]`` (or empty) and records each crop it read."""

    def __init__(self, texts=None):
        self.texts = texts or {}
        self.seen: list[tuple[int, int, int, int]] = []

    def recognize(self, region: Region) -> Region:
        box = region.bbox.to_tuple()
        self.seen.append(box)
        region.text = self.texts.get(box, "")
        region.status = "ok"
        return region


def _texts(layout):
    return [r.text for r in layout.regions]


# ---------------------------------------------------------------------------
# Pass 1: lossless dedup
# ---------------------------------------------------------------------------

def test_near_duplicate_keeps_the_clean_read():
    out = RegionRepair().repair(_layout([
        _r(0, 0, 100, 100, "hello world", status="error"),
        _r(2, 2, 100, 100, "hello world"),
    ]))
    assert len(out.regions) == 1 and out.regions[0].status == "ok"


def test_near_duplicate_folds_variant_tokens_into_keeper():
    regions = [_r(0, 0, 100, 100, "alpha bravo charlie"),
               _r(2, 2, 100, 100, "alpha bravo delta")]
    out = RegionRepair().repair(_layout(regions))
    assert len(out.regions) == 1
    assert "charlie" in out.regions[0].text and "delta" in out.regions[0].text

    out = RegionRepair(lossless=False).repair(_layout(
        [_r(0, 0, 100, 100, "alpha bravo charlie"), _r(2, 2, 100, 100, "alpha bravo delta")]))
    assert _texts(out) == ["alpha bravo charlie"]


def test_strict_substring_inner_region_dropped():
    out = RegionRepair().repair(_layout([
        _r(0, 0, 200, 400, "First para. Second para."),
        _r(0, 0, 200, 100, "first para"),
    ]))
    assert _texts(out) == ["First para. Second para."]


def test_fuzzy_overlap_is_not_dropped():
    out = RegionRepair().repair(_layout([
        _r(0, 0, 200, 400, "First para. Second para."),
        _r(0, 0, 200, 100, "first pora"),          # OCR variant, not a substring
    ]))
    assert len(out.regions) == 2


# ---------------------------------------------------------------------------
# Pass 2: container split
# ---------------------------------------------------------------------------

# A full-column read (A) whose two paragraph reads (B, C) cover 800 of its
# 1000px; the gap at y=400..500 holds text only the container saw.
CONTAINER = (0, 0, 200, 1000, "aaaa bbbb cccc")
INNER_B = (0, 0, 200, 400, "aaaa eeee")
INNER_C = (0, 500, 200, 900, "bbbb ffff")
GAP_STRIP = (0, 400, 200, 500)


def test_container_split_recovers_gap_and_drops_container():
    rec = FakeRecognizer({GAP_STRIP: "cccc"})
    out = RegionRepair(rec).repair(_layout([_r(*CONTAINER), _r(*INNER_B), _r(*INNER_C)]))
    assert _texts(out) == ["aaaa eeee", "bbbb ffff", "cccc"]
    assert out.regions[2].bbox.to_tuple() == GAP_STRIP
    # Both uncovered strips (the gap and the 900..1000 tail) were re-read.
    assert set(rec.seen) == {GAP_STRIP, (0, 900, 200, 1000)}


# iou_merge=1.01 turns off pass 3 so these isolate the split; see
# test_kept_container_is_then_merged_with_its_paragraphs for the interaction.
NO_MERGE = dict(iou_merge=1.01)


def test_container_with_unique_tokens_is_kept():
    rec = FakeRecognizer({GAP_STRIP: "cccc"})
    out = RegionRepair(rec, **NO_MERGE).repair(_layout([
        _r(*CONTAINER[:4], "aaaa bbbb cccc zzzz"), _r(*INNER_B), _r(*INNER_C)]))
    assert "aaaa bbbb cccc zzzz" in _texts(out)
    assert len(out.regions) == 3


def test_container_split_needs_a_recognizer():
    out = RegionRepair(**NO_MERGE).repair(_layout([_r(*CONTAINER), _r(*INNER_B), _r(*INNER_C)]))
    assert _texts(out) == ["aaaa bbbb cccc", "aaaa eeee", "bbbb ffff"]


def test_kept_container_is_then_merged_with_its_paragraphs():
    """Known behavior: a container that survives pass 2 overlaps its paragraph
    reads enough (IoU 0.4 >= iou_merge 0.3) for pass 3 to treat them as ad
    fragments, so the merged text repeats the paragraphs."""
    out = RegionRepair().repair(_layout([_r(*CONTAINER), _r(*INNER_B), _r(*INNER_C)]))
    assert _texts(out) == ["aaaa bbbb cccc\naaaa eeee\nbbbb ffff"]


# ---------------------------------------------------------------------------
# Pass 3: fragmented-ad merge
# ---------------------------------------------------------------------------

FRAG_1 = (0, 0, 300, 300, "sale today")
FRAG_2 = (50, 50, 350, 350, "great sale")
UNION = (0, 0, 350, 350)


def test_fragments_merged_and_reread_once():
    rec = FakeRecognizer({UNION: "great sale today"})
    out = RegionRepair(rec).repair(_layout([_r(*FRAG_1), _r(*FRAG_2)]))
    assert len(out.regions) == 1
    assert out.regions[0].bbox.to_tuple() == UNION
    assert out.regions[0].text == "great sale today"


def test_merge_appends_member_text_the_reread_missed():
    rec = FakeRecognizer({UNION: "sale"})
    out = RegionRepair(rec).repair(_layout([_r(*FRAG_1), _r(*FRAG_2)]))
    assert len(out.regions) == 1
    for word in ("today", "great"):
        assert word in out.regions[0].text


def test_merge_without_recognizer_concatenates_members():
    out = RegionRepair().repair(_layout([_r(*FRAG_1), _r(*FRAG_2)]))
    assert _texts(out) == ["sale today\ngreat sale"]


def test_merge_skipped_when_union_would_swallow_a_sibling():
    rec = FakeRecognizer({UNION: "great sale today"})
    sibling = _r(100, 100, 200, 200, "other text")
    out = RegionRepair(rec).repair(_layout([_r(*FRAG_1), _r(*FRAG_2), sibling]))
    assert len(out.regions) == 3 and UNION not in rec.seen


def test_merge_skipped_when_union_covers_most_of_the_page():
    rec = FakeRecognizer({UNION: "great sale today"})
    out = RegionRepair(rec).repair(_layout([_r(*FRAG_1), _r(*FRAG_2)], w=400, h=400))
    assert len(out.regions) == 2


# ---------------------------------------------------------------------------
# Contract: non-destructive, ids, layout flags, reading order
# ---------------------------------------------------------------------------

def test_input_layout_is_untouched():
    regions = [_r(0, 0, 100, 100, "alpha bravo charlie"),
               _r(2, 2, 100, 100, "alpha bravo delta")]
    for i, r in enumerate(regions):
        r.id = f"orig{i}"
    layout = _layout(regions)
    RegionRepair().repair(layout)
    assert layout.regions is regions and len(regions) == 2
    assert [r.text for r in regions] == ["alpha bravo charlie", "alpha bravo delta"]
    assert [r.id for r in regions] == ["orig0", "orig1"]


def test_ids_renumbered_and_flags_preserved():
    out = RegionRepair().repair(_layout(
        [_r(0, 0, 100, 100, "a"), _r(300, 0, 400, 100, "b")],
        ordered=True, lines_detected=True))
    assert [r.id for r in out.regions] == ["r0", "r1"]
    assert out.ordered and out.lines_detected


def test_recovered_strip_inserted_into_detector_order():
    column_2 = (300, 0, 500, 1000, "gggg")
    regions = lambda: [_r(*CONTAINER), _r(*INNER_B), _r(*INNER_C), _r(*column_2)]
    rec = FakeRecognizer({GAP_STRIP: "cccc"})

    ordered = RegionRepair(rec).repair(_layout(regions(), ordered=True))
    assert _texts(ordered) == ["aaaa eeee", "cccc", "bbbb ffff", "gggg"]

    unordered = RegionRepair(rec).repair(_layout(regions()))
    assert _texts(unordered) == ["aaaa eeee", "bbbb ffff", "gggg", "cccc"]
