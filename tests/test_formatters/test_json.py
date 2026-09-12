import json
from newspaper_ocr.formatters.json_fmt import JsonFormatter
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from PIL import Image
import numpy as np

def test_json_formatter():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    line = Line(bbox=BBox(0, 0, 200, 30), image=img, text="Hello", confidence=0.95)
    region = Region(bbox=BBox(0, 0, 200, 100), image=img, label="article", lines=[line], text="Hello")
    layout = PageLayout(image=img, regions=[region], width=200, height=100)
    result = json.loads(JsonFormatter().format(layout))
    assert result["width"] == 200
    assert result["regions"][0]["label"] == "article"
    assert result["regions"][0]["lines"][0]["text"] == "Hello"
    assert result["regions"][0]["lines"][0]["confidence"] == 0.95
    assert result["regions"][0]["lines"][0]["bbox"] == {"x0": 0, "y0": 0, "x1": 200, "y1": 30}


def test_json_formatter_emits_status_and_id():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    ok = Region(bbox=BBox(0, 0, 200, 50), image=img, label="article", text="Hello")
    bad = Region(
        bbox=BBox(0, 50, 200, 100),
        image=img,
        label="article",
        text="[OCR timeout]",
        status="timeout",
    )
    layout = PageLayout(image=img, regions=[ok, bad], width=200, height=100)
    result = json.loads(JsonFormatter().format(layout))

    assert [r["status"] for r in result["regions"]] == ["ok", "timeout"]
    # Ids fall back to position when the pipeline hasn't assigned them.
    assert [r["id"] for r in result["regions"]] == ["r0", "r1"]


def test_json_formatter_keeps_assigned_region_ids():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    region = Region(bbox=BBox(0, 0, 200, 100), image=img, label="article", id="page3-r7")
    layout = PageLayout(image=img, regions=[region], width=200, height=100)
    result = json.loads(JsonFormatter().format(layout))
    assert result["regions"][0]["id"] == "page3-r7"
