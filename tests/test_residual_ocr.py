"""Tests for the residual second-pass OCR (ResidualOcr)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from newspaper_ocr.models import BBox, PageLayout, Region
from newspaper_ocr.residual_ocr import ResidualOcr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page(w: int = 600, h: int = 400) -> np.ndarray:
    """White RGB page as an (H, W, 3) uint8 array."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _ink(arr: np.ndarray, box: tuple[int, int, int, int]) -> None:
    """Paint a solid black rectangle (ink) into the page array, in-place."""
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = 0


def _layout(arr: np.ndarray, regions: list[Region]) -> PageLayout:
    img = Image.fromarray(arr)
    return PageLayout(image=img, regions=regions, width=arr.shape[1],
                      height=arr.shape[0], lines_detected=False)


def _region(box: tuple[int, int, int, int], text: str = "seen") -> Region:
    x0, y0, x1, y1 = box
    return Region(bbox=BBox(x0, y0, x1, y1),
                  image=Image.new("RGB", (1, 1)), label="text", text=text)


class FakeRecognizer:
    """Region recognizer that returns a fixed text and records what it saw."""

    mode = "region"

    def __init__(self, text: str = "recovered", status: str = "ok"):
        self._text = text
        self._status = status
        self.seen: list[BBox] = []

    def recognize(self, region: Region) -> Region:
        self.seen.append(region.bbox)
        region.text = self._text
        region.status = self._status
        return region


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_requires_recognizer():
    with pytest.raises(ValueError):
        ResidualOcr(None)


def test_recovers_uncovered_ink_block():
    """A black block outside every box becomes one recovered region."""
    arr = _page()
    _ink(arr, (40, 40, 260, 140))      # covered by a pass-1 box
    _ink(arr, (320, 220, 560, 340))    # NOT covered -> should be recovered
    layout = _layout(arr, [_region((40, 40, 260, 140))])

    rec = FakeRecognizer(text="RESIDUAL")
    out = ResidualOcr(rec, min_gain_ink=1.0).recover(layout)

    recovered = [r for r in out.regions if r.engine == "residual"]
    assert len(recovered) == 1
    r = recovered[0]
    assert r.text == "RESIDUAL"
    # Recovered box overlaps the uncovered ink, not the covered block.
    assert r.bbox.x0 >= 300 and r.bbox.y0 >= 200


def test_do_no_harm_when_ink_all_covered():
    """When boxes already cover the ink, the pass is a no-op."""
    arr = _page()
    _ink(arr, (40, 40, 560, 340))
    layout = _layout(arr, [_region((30, 30, 570, 350))])

    rec = FakeRecognizer()
    out = ResidualOcr(rec).recover(layout)

    assert not [r for r in out.regions if r.engine == "residual"]
    assert rec.seen == []               # recognizer never invoked
    assert len(out.regions) == 1


def test_blank_page_is_noop():
    arr = _page()                       # no ink at all
    layout = _layout(arr, [])
    rec = FakeRecognizer()
    out = ResidualOcr(rec).recover(layout)
    assert out.regions == []
    assert rec.seen == []


def test_specks_are_ignored():
    """A tiny uncovered blob below the size floor is not recovered."""
    arr = _page()
    _ink(arr, (300, 200, 308, 206))     # ~8x6 speck
    layout = _layout(arr, [])
    rec = FakeRecognizer()
    out = ResidualOcr(rec, min_block_area=1500).recover(layout)
    assert not [r for r in out.regions if r.engine == "residual"]


def test_empty_reads_are_dropped():
    """Blocks the recognizer returns empty text for are not added."""
    arr = _page()
    _ink(arr, (320, 220, 560, 340))
    layout = _layout(arr, [])
    rec = FakeRecognizer(text="   ")    # whitespace only
    out = ResidualOcr(rec, min_gain_ink=1.0).recover(layout)
    assert not [r for r in out.regions if r.engine == "residual"]


def test_recovered_region_placed_in_reading_order():
    """Recovered top region should sort before a lower pass-1 region."""
    arr = _page()
    _ink(arr, (40, 40, 560, 120))       # top strip, uncovered -> recovered
    layout = _layout(arr, [_region((40, 250, 560, 340), text="bottom")])

    rec = FakeRecognizer(text="TOP")
    out = ResidualOcr(rec, min_gain_ink=1.0).recover(layout)

    texts = [r.text for r in out.regions]
    assert "TOP" in texts and "bottom" in texts
    assert texts.index("TOP") < texts.index("bottom")


class LineStyleRecognizer:
    """Line recognizer that OCRs a region crop via recognize_region (like Tesseract)."""

    mode = "region"

    def __init__(self, text="from_region"):
        self._text = text
        self.region_calls = 0

    def recognize_region(self, region: Region) -> Region:
        self.region_calls += 1
        region.text = self._text
        region.status = "ok"
        return region


def test_uses_recognize_region_when_available():
    """A line recognizer exposing recognize_region reads residual blocks too."""
    arr = _page()
    _ink(arr, (320, 220, 560, 340))
    layout = _layout(arr, [])
    rec = LineStyleRecognizer(text="LINE_ENGINE")
    out = ResidualOcr(rec, min_gain_ink=1.0).recover(layout)
    recovered = [r for r in out.regions if r.engine == "residual"]
    assert rec.region_calls >= 1
    assert recovered and recovered[0].text == "LINE_ENGINE"


def test_converges_within_max_passes():
    """A recognizer that never satisfies the gate must still terminate."""
    arr = _page()
    _ink(arr, (40, 40, 560, 340))       # lots of uncovered ink
    layout = _layout(arr, [])
    rec = FakeRecognizer(text="X")
    resid = ResidualOcr(rec, max_passes=3, min_gain_ink=0.0,
                        run_if_uncovered_ink=0.0)
    out = resid.recover(layout)         # must return, not loop forever
    passes = [a for a in resid.actions if a[0] == "pass"]
    assert len(passes) <= 3
    assert out is not None
