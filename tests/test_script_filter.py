"""Tests for dropping CJK hallucinations on Latin-script pages."""

from __future__ import annotations

from PIL import Image

from newspaper_ocr.models import BBox, PageLayout, Region
from newspaper_ocr.script_filter import drop_cjk_hallucinations


def _layout(*texts):
    regions = [Region(bbox=BBox(0, 10 * i, 100, 10 * i + 10), image=None, label="text", text=t)
               for i, t in enumerate(texts)]
    return PageLayout(image=Image.new("RGB", (100, 100)), regions=regions)


STORY = "The Afro-American Council met in Washington on Tuesday evening."


def test_mostly_cjk_region_is_blanked():
    layout = drop_cjk_hallucinations(_layout(STORY, "(1)基因通过控制 通过控制", "信"))
    texts = [r.text for r in layout.regions]
    assert texts == [STORY, "", ""]
    assert [r.status for r in layout.regions] == ["ok", "hallucination", "hallucination"]
    assert layout.regions[1].text_primary == "(1)基因通过控制 通过控制"


def test_stray_cjk_in_latin_text_is_stripped():
    layout = drop_cjk_hallucinations(_layout(STORY, "Mr. Fortune 表 spoke at length tonight."))
    assert layout.regions[1].text == "Mr. Fortune spoke at length tonight."
    assert layout.regions[1].status == "ok"


def test_cjk_page_is_left_alone():
    layout = drop_cjk_hallucinations(_layout("基因通过控制蛋白质的合成", "表 Table 1"))
    assert [r.text for r in layout.regions] == ["基因通过控制蛋白质的合成", "表 Table 1"]


def test_latin_page_without_cjk_is_unchanged():
    layout = drop_cjk_hallucinations(_layout(STORY, "Café prices, 5¢."))
    assert [r.text for r in layout.regions] == [STORY, "Café prices, 5¢."]
    assert all(r.text_primary == "" for r in layout.regions)
