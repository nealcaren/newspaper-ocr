"""Tests for the AsYoloDetector."""

from pathlib import Path

import pytest
from PIL import Image
import numpy as np

from newspaper_ocr.detectors.as_yolo import (
    LAYOUT_INPUT_SIZE,
    LINE_INPUT_SIZE,
    AsYoloDetector,
    _letterbox,
    _model_input_spec,
    _resolve_model_path,
    _run_line_detection,
    _xywh2xyxy,
)
from newspaper_ocr.models import PageLayout


# ---------------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------------

# Models resolve from the local cache, then the Hugging Face Hub
# (NealCaren/american-stories-onnx). Skip the model-backed tests only when the
# layout model can't be obtained at all (e.g. offline with a cold cache).
def _models_available() -> bool:
    try:
        _resolve_model_path(None, "layout_model_new.onnx", None)
        return True
    except Exception:
        return False


_MODELS_AVAILABLE = _models_available()

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
        # models resolve from cache / the Hugging Face Hub (no local path needed)
        return AsYoloDetector()

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


# ---------------------------------------------------------------------------
# Model input size (no models needed)
# ---------------------------------------------------------------------------


class _FakeInput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeSession:
    """Stands in for an InferenceSession, recording what it was fed."""

    def __init__(self, shape, name="images"):
        self._inputs = [_FakeInput(name, shape)]
        self.fed = []

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, input_feed):
        self.fed.append(next(iter(input_feed.values())).shape)
        raise _StopInference


class _StopInference(Exception):
    """Raised by the fake session once it has recorded its input."""


class TestModelInputSpec:
    def test_reads_the_size_from_the_model(self):
        name, size = _model_input_spec(_FakeSession([1, 3, 640, 640]), LAYOUT_INPUT_SIZE)
        assert (name, size) == ("images", 640)

    def test_layout_and_line_models_have_different_sizes(self):
        """The published models differ — 1280 and 640 — so this can't be shared."""
        _, layout = _model_input_spec(_FakeSession([1, 3, 1280, 1280]), LAYOUT_INPUT_SIZE)
        _, line = _model_input_spec(_FakeSession([1, 3, 640, 640]), LINE_INPUT_SIZE)
        assert (layout, line) == (1280, 640)

    @pytest.mark.parametrize(
        "shape",
        [
            [1, 3, "height", "width"],  # dynamic axes
            [1, 3, -1, -1],  # unspecified
            [1, 3, 640, 800],  # non-square
        ],
    )
    def test_falls_back_when_the_model_does_not_pin_a_square_size(self, shape):
        _, size = _model_input_spec(_FakeSession(shape), LINE_INPUT_SIZE)
        assert size == LINE_INPUT_SIZE

    def test_input_name_still_comes_from_the_model(self):
        name, _ = _model_input_spec(_FakeSession([1, 3, 640, 640], name="input0"), 640)
        assert name == "input0"


class TestLineDetectionInputSize:
    """Regression: line inference was fed 1280 while the model wanted 640.

    The existing detect() tests never caught it because noise and blank pages
    produce no article regions, so line inference was never reached.
    """

    def _crops(self):
        return [("article", (0, 0, 200, 400), Image.new("RGB", (200, 400), "white"))]

    def test_feeds_the_size_the_model_asked_for(self):
        session = _FakeSession([1, 3, 640, 640])
        with pytest.raises(_StopInference):
            _run_line_detection(session, "images", self._crops(), size=640)
        assert session.fed[0][-2:] == (640, 640)

    def test_a_different_model_size_is_honoured(self):
        session = _FakeSession([1, 3, 1280, 1280])
        with pytest.raises(_StopInference):
            _run_line_detection(session, "images", self._crops(), size=1280)
        assert session.fed[0][-2:] == (1280, 1280)


@pytest.mark.skipif(not _MODELS_AVAILABLE, reason="ONNX models not found")
class TestDetectorAdoptsModelSizes:
    def test_sizes_come_from_the_loaded_models(self):
        detector = AsYoloDetector()
        layout_shape = detector._layout_session.get_inputs()[0].shape
        line_shape = detector._line_session.get_inputs()[0].shape

        assert detector._layout_input_size == layout_shape[-1]
        assert detector._line_input_size == line_shape[-1]
