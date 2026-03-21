from newspaper_ocr.formatters.hocr import HocrFormatter
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from PIL import Image
import numpy as np

def test_hocr_has_structure():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    line = Line(bbox=BBox(10, 20, 190, 50), image=img, text="Hello world")
    region = Region(bbox=BBox(0, 0, 200, 100), image=img, label="article", lines=[line], text="Hello world")
    layout = PageLayout(image=img, regions=[region], width=200, height=100)
    result = HocrFormatter().format(layout)
    assert 'class="ocr_page"' in result
    assert 'class="ocr_carea"' in result
    assert 'class="ocr_line"' in result
    assert "Hello world" in result
    assert "bbox 10 20 190 50" in result

def test_hocr_escapes_html():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    line = Line(bbox=BBox(0, 0, 200, 30), image=img, text="A & B < C")
    region = Region(bbox=BBox(0, 0, 200, 100), image=img, label="article", lines=[line])
    layout = PageLayout(image=img, regions=[region], width=200, height=100)
    result = HocrFormatter().format(layout)
    assert "A &amp; B &lt; C" in result
