import pytest
from newspaper_ocr import Pipeline


def test_pipeline_from_strings():
    try:
        pipe = Pipeline(detector="as_yolo", recognizer="tesseract", output="text")
    except (ImportError, KeyError):
        pytest.skip("Required backends not available")
    assert pipe.detector is not None
    assert pipe.recognizer is not None
    assert pipe.formatter is not None


def test_pipeline_default():
    try:
        pipe = Pipeline()
    except (ImportError, KeyError):
        pytest.skip("Default backends not available")
    assert pipe.detector is not None


def test_pipeline_with_layout_processing_disabled():
    try:
        pipe = Pipeline(layout_processing=False)
    except (ImportError, KeyError):
        pytest.skip("Required backends not available")
    assert not pipe.layout_processor.enabled
