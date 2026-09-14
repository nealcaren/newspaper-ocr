"""Tests for the recognizer return contract and the crop helper."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from newspaper_ocr.models import Line, Region
from newspaper_ocr.recognizers.base import (
    LineRecognizer,
    RegionRecognizer,
    recognize_crop,
)


def _img(w: int = 40, h: int = 20) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


class _OnContract(RegionRecognizer):
    """What the contract asks for: mutate the region, return it."""

    def recognize(self, region: Region) -> Region:
        region.text = "clean text"
        region.status = "ok"
        return region


class _ReturnsTuple:
    """The bug the contract exists to catch: ``(text, status)`` instead of a Region."""

    def recognize(self, region: Region):
        return "clean text", "repetition"


class _ReturnsString:
    def recognize(self, region: Region):
        return "clean text"


class _ReturnsNone:
    """Mutates in place and returns nothing."""

    def recognize(self, region: Region):
        region.text = "clean text"
        region.status = "timeout"


class _RegionFallback:
    """A line recognizer with the region fallback the pipeline uses."""

    def recognize_region(self, region: Region) -> Region:
        region.text = "from recognize_region"
        return region


class _LineOnly(LineRecognizer):
    def recognize(self, line: Line) -> Line:
        return line


def test_region_recognizer_returns_str_text():
    text, status = recognize_crop(_OnContract(), _img())
    assert (text, status) == ("clean text", "ok")


def test_tuple_return_is_normalized_to_str():
    """Storing the raw tuple as Region.text is what poisons page JSON with
    list-typed text; the helper unpacks it instead."""
    text, status = recognize_crop(_ReturnsTuple(), _img())
    assert text == "clean text"
    assert status == "repetition"
    assert isinstance(text, str)


def test_bare_string_return_is_accepted():
    assert recognize_crop(_ReturnsString(), _img()) == ("clean text", "ok")


def test_in_place_mutation_is_read_back_off_the_region():
    assert recognize_crop(_ReturnsNone(), _img()) == ("clean text", "timeout")


def test_recognize_region_is_preferred_when_present():
    text, _ = recognize_crop(_RegionFallback(), _img())
    assert text == "from recognize_region"


def test_crop_region_covers_the_whole_image():
    seen: list[Region] = []

    class _Capture(RegionRecognizer):
        def recognize(self, region: Region) -> Region:
            seen.append(region)
            region.text = ""
            return region

    recognize_crop(_Capture(), _img(80, 30), label="image")
    assert seen[0].bbox.to_tuple() == (0, 0, 80, 30)
    assert seen[0].label == "image"


def test_empty_text_is_a_string_not_none():
    class _ReturnsNoneText(RegionRecognizer):
        def recognize(self, region: Region) -> Region:
            region.text = None
            return region

    text, _ = recognize_crop(_ReturnsNoneText(), _img())
    assert text == ""


def test_nested_container_text_is_refused():
    """There is no sane string for a tuple of tuples — better to raise than to
    write ``str(('...', 'ok'))`` into the page."""

    class _Nested:
        def recognize(self, region: Region):
            return (("clean text", "ok"), "ok")

    with pytest.raises(TypeError):
        recognize_crop(_Nested(), _img())


def test_line_only_recognizer_is_refused():
    with pytest.raises(TypeError):
        recognize_crop(_LineOnly(), _img())
