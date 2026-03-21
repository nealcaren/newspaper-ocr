"""Tests for PaddleX detector."""

import pytest
from unittest.mock import patch, MagicMock

from newspaper_ocr.detectors.paddlex import PaddleXDetector
from newspaper_ocr.detectors.base import Detector


class TestPaddleXDetectorImport:
    """Tests that work without PaddleX installed."""

    def test_class_exists_and_is_detector(self):
        assert issubclass(PaddleXDetector, Detector)

    def test_import_error_gives_helpful_message(self):
        with patch.dict("sys.modules", {"paddlex": None}):
            with pytest.raises(ImportError, match="paddlepaddle"):
                PaddleXDetector()


@pytest.fixture
def has_paddlex():
    try:
        import paddlex  # noqa: F401
    except ImportError:
        pytest.skip("paddlex not installed")


class TestPaddleXDetectorIntegration:
    """Tests that require PaddleX (skipped if not installed)."""

    def test_registry_registration(self, has_paddlex):
        from newspaper_ocr.detectors import DETECTORS

        assert "paddlex" in DETECTORS._entries

    def test_detect_returns_page_layout(self, has_paddlex):
        from PIL import Image
        from newspaper_ocr.models import PageLayout

        detector = PaddleXDetector()
        img = Image.new("RGB", (100, 100), "white")
        layout = detector.detect(img)
        assert isinstance(layout, PageLayout)
