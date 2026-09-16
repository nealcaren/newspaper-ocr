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
