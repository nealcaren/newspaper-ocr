from newspaper_ocr.models import BBox, Line, Region, PageLayout
from PIL import Image
import numpy as np


def test_bbox_creation():
    bbox = BBox(x0=10, y0=20, x1=100, y1=50)
    assert bbox.x0 == 10
    assert bbox.width == 90
    assert bbox.height == 30


def test_bbox_to_tuple():
    bbox = BBox(0, 0, 100, 50)
    assert bbox.to_tuple() == (0, 0, 100, 50)


def test_line_default_text():
    img = Image.fromarray(np.zeros((30, 200, 3), dtype=np.uint8))
    line = Line(bbox=BBox(0, 0, 200, 30), image=img)
    assert line.text == ""
    assert line.confidence == 0.0


def test_region_has_confidence():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    region = Region(bbox=BBox(0, 0, 200, 100), image=img, label="article", confidence=0.85)
    assert region.confidence == 0.85


def test_page_layout_text():
    img = Image.fromarray(np.zeros((500, 400, 3), dtype=np.uint8))
    r1 = Region(bbox=BBox(0, 0, 400, 200), image=img, label="article", lines=[], text="First paragraph.")
    r2 = Region(bbox=BBox(0, 200, 400, 500), image=img, label="article", lines=[], text="Second paragraph.")
    layout = PageLayout(image=img, regions=[r1, r2], width=400, height=500)
    assert layout.text == "First paragraph.\n\nSecond paragraph."


def test_page_layout_skips_empty_regions():
    img = Image.fromarray(np.zeros((500, 400, 3), dtype=np.uint8))
    r1 = Region(bbox=BBox(0, 0, 400, 200), image=img, label="article", lines=[], text="Content")
    r2 = Region(bbox=BBox(0, 200, 400, 500), image=img, label="photograph", lines=[], text="")
    layout = PageLayout(image=img, regions=[r1, r2], width=400, height=500)
    assert layout.text == "Content"
