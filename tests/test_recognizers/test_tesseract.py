import subprocess
import pytest
from newspaper_ocr.recognizers.tesseract import TesseractRecognizer
from newspaper_ocr.models import BBox, Line, Region
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


def _make_region_image(lines=None):
    """Create a multi-line region image."""
    if lines is None:
        lines = ["Hello World", "Second line", "Third line"]
    img = Image.new("RGB", (400, 40 * len(lines) + 20), "white")
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((10, 10 + i * 40), text, fill="black")
    return img


# --- Line mode tests (existing) ---

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


# --- Mode parameter tests ---

def test_tesseract_default_mode_is_region():
    """TesseractRecognizer defaults to region mode."""
    rec = TesseractRecognizer()
    assert rec.mode == "region"


def test_tesseract_mode_region():
    """TesseractRecognizer accepts region mode."""
    rec = TesseractRecognizer(mode="region")
    assert rec.mode == "region"


def test_tesseract_invalid_mode():
    """TesseractRecognizer rejects invalid mode."""
    with pytest.raises(ValueError, match="mode must be"):
        TesseractRecognizer(mode="invalid")


# --- Region mode tests ---

def test_tesseract_recognize_region(has_tesseract):
    """Region mode recognizes multi-line text block."""
    img = _make_region_image(["Hello World", "Second line"])
    region = Region(
        bbox=BBox(0, 0, 400, 100),
        image=img,
        label="text",
    )
    rec = TesseractRecognizer(mode="region")
    result = rec.recognize_region(region)
    assert len(result.text) > 0
    assert result.confidence > 0


def test_tesseract_recognize_region_has_method(has_tesseract):
    """Both line and region mode TesseractRecognizer have recognize_region."""
    rec_line = TesseractRecognizer(mode="line")
    rec_region = TesseractRecognizer(mode="region")
    assert hasattr(rec_line, "recognize_region")
    assert hasattr(rec_region, "recognize_region")
