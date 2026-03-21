"""Tests for GLM-OCR recognizer."""

import pytest
from unittest.mock import patch

from newspaper_ocr.recognizers.glm_ocr import GlmOcrRecognizer
from newspaper_ocr.recognizers.base import RegionRecognizer


class TestGlmOcrImport:
    """Tests that work without GLM-OCR dependencies installed."""

    def test_class_exists_and_is_region_recognizer(self):
        assert issubclass(GlmOcrRecognizer, RegionRecognizer)

    def test_import_error_local_mode(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                GlmOcrRecognizer(mode="local")

    def test_import_error_api_mode(self):
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx"):
                GlmOcrRecognizer(mode="api")

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            GlmOcrRecognizer(mode="bogus")


class TestRepetitionDetection:
    """Test the static repetition helpers (no deps needed)."""

    def test_no_repetition(self):
        assert not GlmOcrRecognizer._has_repetition("Hello world this is normal text.")

    def test_detects_repetition(self):
        repeated = "The quick brown fox " * 10
        assert GlmOcrRecognizer._has_repetition(repeated)

    def test_truncate_repetition(self):
        repeated = "Hello world " * 5
        result = GlmOcrRecognizer._truncate_repetition(repeated)
        # Should return just the first occurrence, stripped
        assert len(result) < len(repeated)
        assert "Hello world" in result

    def test_truncate_no_repetition(self):
        text = "Unique text here."
        assert GlmOcrRecognizer._truncate_repetition(text) == text


class TestGlmOcrRecognize:
    """Test recognize() with mocked backends."""

    def test_recognize_sets_region_text(self):
        from PIL import Image
        from newspaper_ocr.models import BBox, Region

        region = Region(
            bbox=BBox(0, 0, 100, 50),
            image=Image.new("RGB", (100, 50), "white"),
            label="text",
            lines=[],
        )

        recognizer = GlmOcrRecognizer.__new__(GlmOcrRecognizer)
        recognizer.mode = "api"
        recognizer.max_retries = 0
        recognizer._client = None

        # Mock _recognize_api
        recognizer._recognize_api = lambda img: "Recognized text"

        result = recognizer.recognize(region)
        assert result.text == "Recognized text"

    def test_recognize_handles_exception(self):
        from PIL import Image
        from newspaper_ocr.models import BBox, Region

        region = Region(
            bbox=BBox(0, 0, 100, 50),
            image=Image.new("RGB", (100, 50), "white"),
            label="text",
            lines=[],
        )

        recognizer = GlmOcrRecognizer.__new__(GlmOcrRecognizer)
        recognizer.mode = "api"
        recognizer.max_retries = 0

        recognizer._recognize_api = lambda img: (_ for _ in ()).throw(
            RuntimeError("timeout")
        )

        result = recognizer.recognize(region)
        assert result.text == ""


@pytest.fixture
def has_glm_ocr():
    try:
        from transformers import AutoProcessor  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")


class TestGlmOcrIntegration:
    """Tests that require transformers (skipped if not installed)."""

    def test_registry_registration(self, has_glm_ocr):
        from newspaper_ocr.recognizers import RECOGNIZERS

        assert "glm-ocr" in RECOGNIZERS._entries
