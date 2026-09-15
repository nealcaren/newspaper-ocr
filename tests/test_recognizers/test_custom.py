"""Tests for the callable (bring-your-own-function) recognizer adapter."""

import pytest

from PIL import Image

from newspaper_ocr.models import Region
from newspaper_ocr.pipeline import Pipeline
from newspaper_ocr.recognizers.base import RegionRecognizer
from newspaper_ocr.recognizers.custom import CallableRegionRecognizer, TIMEOUT_TEXT


def _region():
    return Region(bbox=None, image=Image.new("RGB", (40, 15), "white"), label="text")


class TestCallableRegionRecognizer:
    def test_is_region_recognizer(self):
        assert issubclass(CallableRegionRecognizer, RegionRecognizer)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            CallableRegionRecognizer("not callable")

    def test_calls_fn_with_image_and_returns_text(self):
        seen = {}

        def fn(image):
            seen["image"] = image
            return "  HELLO  "

        reg = _region()
        out = CallableRegionRecognizer(fn).recognize(reg)
        assert out.text == "HELLO"
        assert out.status == "ok"
        assert seen["image"] is reg.image

    def test_label_defaults_to_fn_name(self):
        def my_ocr(image):
            return ""

        assert CallableRegionRecognizer(my_ocr).label == "my_ocr"

    def test_error_yields_error_status(self):
        def boom(image):
            raise RuntimeError("nope")

        reg = CallableRegionRecognizer(boom, max_retries=0).recognize(_region())
        assert reg.text == ""
        assert reg.status == "error"

    def test_timeout_yields_timeout_status(self):
        def slow(image):
            raise TimeoutError("slow")

        reg = CallableRegionRecognizer(slow, max_retries=0).recognize(_region())
        assert reg.text == TIMEOUT_TEXT
        assert reg.status == "timeout"

    def test_none_return_is_empty_text(self):
        reg = CallableRegionRecognizer(lambda image: None).recognize(_region())
        assert reg.text == ""
        assert reg.status == "ok"


class TestPipelineAcceptsCallable:
    def test_wraps_plain_callable(self):
        rec = Pipeline._as_recognizer(lambda image: "text")
        assert isinstance(rec, CallableRegionRecognizer)

    def test_passes_instance_through(self):
        inst = CallableRegionRecognizer(lambda image: "x")
        assert Pipeline._as_recognizer(inst) is inst

    def test_duck_typed_object_passes_through(self):
        class Duck:
            def recognize(self, region):
                return region

        d = Duck()
        assert Pipeline._as_recognizer(d) is d

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            Pipeline._as_recognizer(42)
