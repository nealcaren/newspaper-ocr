import importlib.util
from unittest import mock

import pytest
from newspaper_ocr import Pipeline


def _spec(*present):
    """find_spec stub: truthy for module names in *present*, None otherwise."""
    return lambda name: object() if name in present else None


def test_auto_detector_prefers_doclayout_when_available():
    with mock.patch.object(importlib.util, "find_spec",
                           side_effect=_spec("doclayout_yolo", "paddlex")):
        assert Pipeline._resolve_detector_name("auto") == "doclayout_yolo"


def test_auto_detector_prefers_paddlex_when_no_doclayout():
    with mock.patch.object(importlib.util, "find_spec", side_effect=_spec("paddlex")):
        assert Pipeline._resolve_detector_name("auto") == "paddlex"


def test_auto_detector_falls_back_to_as_yolo_with_warning():
    with mock.patch.object(importlib.util, "find_spec", return_value=None):
        with pytest.warns(UserWarning, match="as_yolo"):
            assert Pipeline._resolve_detector_name("auto") == "as_yolo"


def test_explicit_detector_name_passes_through():
    assert Pipeline._resolve_detector_name("as_yolo") == "as_yolo"
    assert Pipeline._resolve_detector_name("paddlex") == "paddlex"


def test_pipeline_from_strings():
    try:
        pipe = Pipeline(detector="as_yolo", recognizer="tesseract", output="text")
    except (ImportError, KeyError, FileNotFoundError):
        pytest.skip("Required backends not available")
    assert pipe.detector is not None
    assert pipe.recognizer is not None
    assert pipe.formatter is not None


def test_pipeline_default():
    try:
        pipe = Pipeline()
    except (ImportError, KeyError, FileNotFoundError):
        pytest.skip("Default backends not available")
    assert pipe.detector is not None


def test_pipeline_with_layout_processing_disabled():
    try:
        pipe = Pipeline(layout_processing=False)
    except (ImportError, KeyError, FileNotFoundError):
        pytest.skip("Required backends not available")
    assert not pipe.layout_processor.enabled


def test_hole_fill_detector_wraps_primary_in_union():
    from PIL import Image
    from newspaper_ocr.detectors.base import Detector
    from newspaper_ocr.detectors.union import UnionDetector
    from newspaper_ocr.models import PageLayout

    class Empty(Detector):
        def detect(self, image):
            return PageLayout(image=image, width=image.size[0], height=image.size[1])

    primary, secondary = Empty(), Empty()
    pipe = Pipeline(detector=primary, hole_fill_detector=secondary,
                    recognizer=lambda img: "", residual_ocr=False)
    assert isinstance(pipe.detector, UnionDetector)
    assert pipe.detector.primary is primary and pipe.detector.secondary is secondary
    assert pipe.analyze(Image.new("RGB", (50, 50), "white")).regions == []

    assert Pipeline(detector=primary, recognizer=lambda img: "",
                    residual_ocr=False).detector is primary


def test_region_recognizer_batch_hook_reads_page_once():
    from PIL import Image
    from newspaper_ocr.detectors.base import Detector
    from newspaper_ocr.models import BBox, PageLayout, Region
    from newspaper_ocr.recognizers.base import RegionRecognizer

    class TwoBoxes(Detector):
        def detect(self, image):
            regions = [Region(bbox=BBox(0, 0, 10, 10), image=image.crop((0, 0, 10, 10)),
                              label="text", confidence=1.0),
                       Region(bbox=BBox(0, 20, 10, 30), image=image.crop((0, 20, 10, 30)),
                              label="text", confidence=1.0)]
            return PageLayout(image=image, regions=regions, width=50, height=50,
                              ordered=True)

    class Batched(RegionRecognizer):
        calls = 0

        def recognize(self, region):
            raise AssertionError("per-region path should not run")

        def recognize_regions(self, page_image, regions):
            Batched.calls += 1
            for i, r in enumerate(regions):
                r.text = f"t{i}"
            return regions

    pipe = Pipeline(detector=TwoBoxes(), recognizer=Batched(), residual_ocr=False)
    layout = pipe.analyze(Image.new("RGB", (50, 50), "white"))
    assert Batched.calls == 1
    assert [r.text for r in layout.regions] == ["t0", "t1"]
