"""Tests for the DocLayout-YOLO detector.

The detect() logic is tested with a fake YOLOv10 result, so it needs neither the
``doclayout-yolo`` package nor a model download.
"""

import pytest
from PIL import Image

from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.detectors.doclayout import DocLayoutYoloDetector
from newspaper_ocr.models import PageLayout


class _Vec:
    """Mimics a 1-row tensor: indexable and supports .tolist()."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [_Vec(xyxy)]
        self.cls = [cls]
        self.conf = [conf]


class _FakeResult:
    names = {0: "title", 1: "plain text", 3: "figure"}

    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Records the predict() call and returns a canned result."""

    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(self._boxes)]


def _detector_with(model):
    det = DocLayoutYoloDetector.__new__(DocLayoutYoloDetector)
    det.model = model
    det.imgsz = 1024
    det.conf = 0.2
    det.device = "cpu"
    return det


class TestDocLayoutImport:
    def test_class_is_a_detector(self):
        assert issubclass(DocLayoutYoloDetector, Detector)

    def test_import_error_is_helpful(self):
        from unittest.mock import patch

        with patch.dict("sys.modules", {"doclayout_yolo": None}):
            with pytest.raises(ImportError, match="doclayout-yolo"):
                DocLayoutYoloDetector()

    def test_registered(self):
        from newspaper_ocr.detectors import DETECTORS

        assert "doclayout_yolo" in DETECTORS._entries

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            DocLayoutYoloDetector(variant="does-not-exist")

    def test_default_variant_is_the_1280_model(self):
        from newspaper_ocr.detectors.doclayout import DEFAULT_VARIANT, VARIANTS

        repo, filename, imgsz = VARIANTS[DEFAULT_VARIANT]
        assert imgsz == 1280
        assert "imgsz1280" in filename


class TestDocLayoutDetect:
    def test_returns_page_layout_with_regions(self):
        model = _FakeModel(
            [
                _FakeBox([10.0, 20.0, 110.0, 60.0], 0, 0.9),  # title
                _FakeBox([5.0, 70.0, 190.0, 290.0], 1, 0.8),  # plain text
            ]
        )
        det = _detector_with(model)
        img = Image.new("RGB", (200, 300), "white")

        layout = det.detect(img)

        assert isinstance(layout, PageLayout)
        assert layout.width == 200 and layout.height == 300
        # Region-only detector: no lines detected.
        assert layout.lines_detected is False
        assert [r.label for r in layout.regions] == ["title", "plain_text"]
        assert layout.regions[0].bbox.to_tuple() == (10, 20, 110, 60)
        assert layout.regions[1].confidence == pytest.approx(0.8)
        # imgsz/conf/device were forwarded to predict().
        assert model.calls[0]["imgsz"] == 1024

    def test_labels_have_spaces_normalized(self):
        model = _FakeModel([_FakeBox([0.0, 0.0, 50.0, 50.0], 1, 0.5)])
        layout = _detector_with(model).detect(Image.new("RGB", (100, 100)))
        assert layout.regions[0].label == "plain_text"

    def test_out_of_bounds_and_degenerate_boxes_are_clamped_or_dropped(self):
        model = _FakeModel(
            [
                _FakeBox([-10.0, -10.0, 500.0, 500.0], 0, 0.9),  # clamped to page
                _FakeBox([50.0, 50.0, 50.0, 90.0], 1, 0.7),  # zero width -> dropped
            ]
        )
        layout = _detector_with(model).detect(Image.new("RGB", (100, 100)))
        assert len(layout.regions) == 1
        assert layout.regions[0].bbox.to_tuple() == (0, 0, 100, 100)
