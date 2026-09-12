"""Tests for the PaddleOCR-VL recognizer.

recognize() is driven with fake model/processor objects and monkeypatched
_recognize_local, so the tests need neither transformers nor a model download.
"""

import pytest
from PIL import Image

from newspaper_ocr.errors import OcrTimeout
from newspaper_ocr.models import Region
from newspaper_ocr.recognizers.base import RegionRecognizer
from newspaper_ocr.recognizers.glm_ocr import TIMEOUT_TEXT
from newspaper_ocr.recognizers.paddleocr_vl import PaddleOcrVlRecognizer


def _region():
    return Region(
        bbox=None, image=Image.new("RGB", (40, 20), "white"), label="plain_text"
    )


def _recognizer(text=None, exc=None, max_retries=1):
    """A recognizer whose _recognize_local yields fixed text or raises."""
    rec = PaddleOcrVlRecognizer.__new__(PaddleOcrVlRecognizer)
    rec.max_retries = max_retries
    rec.repetition_min_len = 20
    rec.repetition_min_reps = 5
    calls = {"n": 0}

    def fake(image):
        calls["n"] += 1
        if exc is not None:
            raise exc
        return text

    rec._recognize_local = fake
    rec._calls = calls
    return rec


class TestPaddleOcrVlBasics:
    def test_is_a_region_recognizer(self):
        assert issubclass(PaddleOcrVlRecognizer, RegionRecognizer)

    def test_registered(self):
        from newspaper_ocr.recognizers import RECOGNIZERS

        assert "paddleocr-vl" in RECOGNIZERS._entries

    def test_import_error_is_helpful(self):
        from unittest.mock import patch

        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers>=5.3"):
                PaddleOcrVlRecognizer()


class TestPaddleOcrVlRecognize:
    def test_clean_text_is_ok(self):
        rec = _recognizer(text="Washington, D. C., March 19.")
        r = rec.recognize(_region())
        assert r.status == "ok"
        assert r.text == "Washington, D. C., March 19."

    def test_timeout_gives_placeholder_and_status(self):
        rec = _recognizer(exc=OcrTimeout("exceeded"), max_retries=0)
        r = rec.recognize(_region())
        assert r.status == "timeout"
        assert r.text == TIMEOUT_TEXT

    def test_non_timeout_error_gives_error_status(self):
        rec = _recognizer(exc=RuntimeError("cuda oom"), max_retries=0)
        r = rec.recognize(_region())
        assert r.status == "error"
        assert r.text == ""

    def test_repetition_is_truncated_after_retries(self):
        loop = "spam spam " * 20
        rec = _recognizer(text=loop, max_retries=1)
        r = rec.recognize(_region())
        assert r.status == "repetition"
        assert len(r.text) < len(loop)
        # one initial attempt + one retry, both looping
        assert rec._calls["n"] == 2

    def test_retry_then_success(self):
        rec = PaddleOcrVlRecognizer.__new__(PaddleOcrVlRecognizer)
        rec.max_retries = 1
        rec.repetition_min_len = 20
        rec.repetition_min_reps = 5
        seq = iter([OcrTimeout("boom"), "recovered text"])

        def fake(image):
            val = next(seq)
            if isinstance(val, Exception):
                raise val
            return val

        rec._recognize_local = fake
        r = rec.recognize(_region())
        assert r.status == "ok"
        assert r.text == "recovered text"
