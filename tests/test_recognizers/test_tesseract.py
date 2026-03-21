import subprocess
import pytest
from newspaper_ocr.recognizers.tesseract import TesseractRecognizer
from newspaper_ocr.models import BBox, Line
from PIL import Image, ImageDraw


@pytest.fixture
def has_tesseract():
    result = subprocess.run(["tesseract", "--version"], capture_output=True)
    if result.returncode != 0:
        pytest.skip("tesseract not installed")


def _make_line_image(text="Hello World"):
    img = Image.new("RGB", (400, 50), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill="black")
    return img


def test_tesseract_recognizes_text(has_tesseract):
    img = _make_line_image("Hello World")
    line = Line(bbox=BBox(0, 0, 400, 50), image=img)
    rec = TesseractRecognizer()
    result = rec.recognize(line)
    assert len(result.text) > 0
    assert result.confidence > 0


def test_tesseract_custom_model(has_tesseract):
    rec = TesseractRecognizer(model="eng")
    img = _make_line_image("Test")
    line = Line(bbox=BBox(0, 0, 400, 50), image=img)
    result = rec.recognize(line)
    assert len(result.text) > 0


def test_tesseract_batch(has_tesseract):
    lines = [
        Line(bbox=BBox(0, 0, 400, 50), image=_make_line_image("First")),
        Line(bbox=BBox(0, 0, 400, 50), image=_make_line_image("Second")),
    ]
    rec = TesseractRecognizer()
    results = rec.recognize_batch(lines)
    assert len(results) == 2
    assert all(r.text for r in results)
