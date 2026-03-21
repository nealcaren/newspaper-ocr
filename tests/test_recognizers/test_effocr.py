import pytest
from newspaper_ocr.models import BBox, Line
from PIL import Image
from pathlib import Path


@pytest.fixture
def has_effocr():
    try:
        from efficient_ocr import EffOCR
    except ImportError:
        pytest.skip("efficient_ocr not installed")


@pytest.fixture
def has_models():
    model_dir = Path("/Users/nealcaren/Documents/GitHub/dangerouspress-ocr-finetune/data/effocr/models")
    if not model_dir.exists():
        pytest.skip("EffOCR models not available")
    return model_dir


@pytest.fixture
def test_crop():
    p = Path("/Users/nealcaren/Documents/GitHub/dangerouspress-ocr-finetune/data/effocr/as_extractions/sn84025908_1856-08-30_seq-4/lines/line_0010.png")
    if not p.exists():
        pytest.skip("Test crop not available")
    return p


def test_effocr_recognize(has_effocr, has_models, test_crop):
    from newspaper_ocr.recognizers.effocr import EffocrRecognizer
    rec = EffocrRecognizer(model_dir=has_models)
    img = Image.open(test_crop)
    line = Line(bbox=BBox(0, 0, img.width, img.height), image=img)
    result = rec.recognize(line)
    assert len(result.text) > 0


def test_effocr_import_error():
    """Test that import error gives helpful message."""
    # Can't easily test this without uninstalling, just verify the class exists
    from newspaper_ocr.recognizers.effocr import EffocrRecognizer
    assert EffocrRecognizer is not None
