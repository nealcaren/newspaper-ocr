"""Tests for cutting VLM refusals and image descriptions."""

from __future__ import annotations

from PIL import Image

from newspaper_ocr.models import BBox, PageLayout, Region
from newspaper_ocr.refusal_filter import drop_refusals, strip_refusals


def _layout(*texts):
    regions = [Region(bbox=BBox(0, 10 * i, 100, 10 * i + 10), image=None, label="text", text=t)
               for i, t in enumerate(texts)]
    return PageLayout(image=Image.new("RGB", (100, 100)), regions=regions)


STORY = "The Afro-American Council met in Washington on Tuesday evening."


def test_refusal_region_is_blanked():
    layout = drop_refusals(_layout(STORY, "There is no legible text in this image.",
                                   "I’m sorry, but I can’t read the text in this image."))
    assert [r.text for r in layout.regions] == [STORY, "", ""]
    assert [r.status for r in layout.regions] == ["ok", "hallucination", "hallucination"]
    assert layout.regions[1].text_primary == "There is no legible text in this image."


def test_refusal_sentence_is_cut_from_real_text():
    text = "The image has been rotated 90 degrees clockwise.\nMr. Fortune spoke at length."
    layout = drop_refusals(_layout(text))
    assert layout.regions[0].text == "Mr. Fortune spoke at length."
    assert layout.regions[0].status == "ok"


def test_transcription_preamble_is_cut():
    text, cut = strip_refusals("The text in the image is:\nLOST, a gray mare.")
    assert cut and text == "LOST, a gray mare."


def test_newspaper_prose_about_images_is_kept():
    for text in ["We are told that man was made in the image of God; that, although finite.",
                 "though the image of his Creator has been lost by transgression",
                 "I cannot see in your Constitution any grounds for the claim.",
                 "I am sorry to say the text of the bill was not printed."]:
        assert strip_refusals(text) == (text, False)


def test_page_without_refusals_is_unchanged():
    layout = drop_refusals(_layout(STORY, "Café prices, 5¢."))
    assert [r.text for r in layout.regions] == [STORY, "Café prices, 5¢."]
    assert all(r.text_primary == "" for r in layout.regions)
