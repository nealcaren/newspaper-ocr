from newspaper_ocr.formatters.text import TextFormatter
from newspaper_ocr.models import BBox, Region, PageLayout
from PIL import Image
import numpy as np

def _make_layout():
    img = Image.fromarray(np.zeros((500, 400, 3), dtype=np.uint8))
    r1 = Region(bbox=BBox(0, 0, 400, 200), image=img, label="headline", lines=[], text="BIG NEWS TODAY")
    r2 = Region(bbox=BBox(0, 200, 400, 500), image=img, label="article", lines=[], text="Something happened.")
    return PageLayout(image=img, regions=[r1, r2], width=400, height=500)

def test_text_formatter():
    result = TextFormatter().format(_make_layout())
    assert "BIG NEWS TODAY" in result
    assert "Something happened." in result
