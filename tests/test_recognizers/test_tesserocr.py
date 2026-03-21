import pytest
from newspaper_ocr.models import BBox, Line, Region
from PIL import Image, ImageDraw


def _has_tesserocr():
    try:
        import tesserocr  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_tesserocr(), reason="tesserocr not installed")


def _make_line_image(text="Hello World"):
    img = Image.new("RGB", (400, 50), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill="black")
    return img


def _make_region_image(lines=None):
    if lines is None:
        lines = ["Hello World", "Second line", "Third line"]
    img = Image.new("RGB", (400, 40 * len(lines) + 20), "white")
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((10, 10 + i * 40), text, fill="black")
    return img


def test_tesserocr_line_mode():
    from newspaper_ocr.recognizers.tesserocr_backend import TesserocrRecognizer

    rec = TesserocrRecognizer(mode="line")
    img = _make_line_image("Hello World")
    line = Line(bbox=BBox(0, 0, 400, 50), image=img)
    result = rec.recognize(line)
    assert len(result.text) > 0
    assert result.confidence > 0


def test_tesserocr_region_mode():
    from newspaper_ocr.recognizers.tesserocr_backend import TesserocrRecognizer

    rec = TesserocrRecognizer(mode="region")
    img = _make_region_image(["Hello World", "Second line"])
    region = Region(
        bbox=BBox(0, 0, 400, 100),
        image=img,
        label="text",
    )
    result = rec.recognize_region(region)
    assert len(result.text) > 0
    assert result.confidence > 0


def test_tesserocr_invalid_mode():
    from newspaper_ocr.recognizers.tesserocr_backend import TesserocrRecognizer

    with pytest.raises(ValueError, match="mode must be"):
        TesserocrRecognizer(mode="invalid")


def test_tesserocr_registered():
    from newspaper_ocr.recognizers import RECOGNIZERS

    assert "tesserocr" in RECOGNIZERS.list()


def test_tesserocr_batch():
    from newspaper_ocr.recognizers.tesserocr_backend import TesserocrRecognizer

    rec = TesserocrRecognizer(mode="line")
    lines = [
        Line(bbox=BBox(0, 0, 400, 50), image=_make_line_image("First")),
        Line(bbox=BBox(0, 0, 400, 50), image=_make_line_image("Second")),
    ]
    results = rec.recognize_batch(lines)
    assert len(results) == 2
    assert all(r.text for r in results)
