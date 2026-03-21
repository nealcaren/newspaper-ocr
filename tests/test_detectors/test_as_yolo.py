"""Tests for the AsYoloDetector."""

from pathlib import Path

import pytest
from PIL import Image
import numpy as np

from newspaper_ocr.detectors.as_yolo import AsYoloDetector, _letterbox, _xywh2xyxy
from newspaper_ocr.models import PageLayout


# ---------------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------------

_DROPBOX_DIR = Path(
    "/Users/nealcaren/Dropbox/american-stories/american_stories_models"
)
_MODELS_AVAILABLE = (
    (_DROPBOX_DIR / "layout_model_new.onnx").is_file()
    and (_DROPBOX_DIR / "line_model_new.onnx").is_file()
)

_JP2_PATH = Path(
    "/Volumes/Lightning/chronicling-america/loc_downloads/sn84025908/1856-08-30/seq-4.jp2"
)
_JP2_AVAILABLE = _JP2_PATH.is_file()


# ---------------------------------------------------------------------------
# Unit tests for helpers (no models needed)
# ---------------------------------------------------------------------------


class TestLetterbox:
    def test_square_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result, ratios, padding = _letterbox(img, (640, 640))
        assert result.shape == (640, 640, 3)

    def test_tall_image(self):
        img = np.zeros((1000, 500, 3), dtype=np.uint8)
        result, ratios, padding = _letterbox(img, (640, 640))
        assert result.shape[0] == 640
        assert result.shape[1] == 640

    def test_wide_image(self):
        img = np.zeros((500, 1000, 3), dtype=np.uint8)
        result, ratios, padding = _letterbox(img, (640, 640))
        assert result.shape[0] == 640
        assert result.shape[1] == 640


class TestXywh2xyxy:
    def test_basic_conversion(self):
        import torch

        boxes = torch.tensor([[50.0, 50.0, 20.0, 30.0]])
        result = _xywh2xyxy(boxes)
        assert result[0, 0].item() == pytest.approx(40.0)
        assert result[0, 1].item() == pytest.approx(35.0)
        assert result[0, 2].item() == pytest.approx(60.0)
        assert result[0, 3].item() == pytest.approx(65.0)


# ---------------------------------------------------------------------------
# Integration tests requiring ONNX models
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MODELS_AVAILABLE, reason="ONNX models not found")
class TestAsYoloDetectorWithModels:
    @pytest.fixture(scope="class")
    def detector(self):
        return AsYoloDetector(model_dir=_DROPBOX_DIR)

    def test_random_noise_image(self, detector):
        """Random noise should not crash; may produce zero or some detections."""
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, (800, 600, 3), dtype=np.uint8)
        img = Image.fromarray(noise, "RGB")
        layout = detector.detect(img)

        assert isinstance(layout, PageLayout)
        assert layout.width == 600
        assert layout.height == 800

    def test_blank_white_image(self, detector):
        """A blank white image should produce no or few detections."""
        img = Image.new("RGB", (1000, 1500), color=(255, 255, 255))
        layout = detector.detect(img)
        assert isinstance(layout, PageLayout)

    def test_grayscale_input(self, detector):
        """Grayscale images should be handled without error."""
        img = Image.new("L", (800, 1200), color=200)
        layout = detector.detect(img)
        assert isinstance(layout, PageLayout)

    @pytest.mark.skipif(not _JP2_AVAILABLE, reason="JP2 test file not available")
    def test_real_newspaper_jp2(self, detector):
        """Real newspaper scan should detect many regions and lines."""
        img = Image.open(_JP2_PATH).convert("RGB")
        layout = detector.detect(img)

        assert isinstance(layout, PageLayout)
        assert layout.width == img.width
        assert layout.height == img.height

        total_lines = sum(len(r.lines) for r in layout.regions)
        print(f"\nReal JP2 results: {len(layout.regions)} regions, {total_lines} lines")

        # Print breakdown by label
        from collections import Counter

        label_counts = Counter(r.label for r in layout.regions)
        for label, count in label_counts.most_common():
            lines_in_label = sum(
                len(r.lines) for r in layout.regions if r.label == label
            )
            print(f"  {label}: {count} regions, {lines_in_label} lines")

        assert len(layout.regions) >= 50, (
            f"Expected 50+ regions, got {len(layout.regions)}"
        )
        assert total_lines >= 500, (
            f"Expected 500+ lines, got {total_lines}"
        )

        # Verify line crops are valid PIL images
        for region in layout.regions:
            for line in region.lines:
                assert isinstance(line.image, Image.Image)
                assert line.image.size[0] > 0
                assert line.image.size[1] > 0
                assert line.bbox.x0 < line.bbox.x1
                assert line.bbox.y0 < line.bbox.y1
