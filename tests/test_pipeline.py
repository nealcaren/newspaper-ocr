from newspaper_ocr.pipeline import Pipeline
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.recognizers.base import LineRecognizer
from newspaper_ocr.formatters.base import Formatter
from PIL import Image
import numpy as np


class MockDetector(Detector):
    def detect(self, image):
        w, h = image.size
        line_img = image.crop((0, 0, w, 30))
        line = Line(bbox=BBox(0, 0, w, 30), image=line_img)
        region = Region(bbox=BBox(0, 0, w, h), image=image, label="article", lines=[line])
        return PageLayout(image=image, regions=[region], width=w, height=h)


class MockRecognizer(LineRecognizer):
    def recognize(self, line):
        line.text = "mock text"
        line.confidence = 0.99
        return line


class MockFormatter(Formatter):
    def format(self, layout):
        return layout.text


def test_pipeline_end_to_end():
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=MockRecognizer(),
        formatter=MockFormatter(),
    )
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    result = pipe.run(img)
    assert "mock text" in result


def test_pipeline_ocr_from_path(tmp_path):
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    img_path = tmp_path / "test.png"
    img.save(str(img_path))
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=MockRecognizer(),
        formatter=MockFormatter(),
    )
    result = pipe.ocr(str(img_path))
    assert "mock text" in result


def test_pipeline_batch(tmp_path):
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    paths = []
    for i in range(3):
        p = tmp_path / f"test_{i}.png"
        img.save(str(p))
        paths.append(str(p))
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=MockRecognizer(),
        formatter=MockFormatter(),
    )
    results = pipe.ocr_batch(paths)
    assert len(results) == 3
    assert all("mock text" in r for r in results)
