"""Tests for the MinerU2.5 detector and recognizer, with a fake client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("mineru_vl_utils")

from newspaper_ocr import _mineru  # noqa: E402
from newspaper_ocr.models import BBox, Region  # noqa: E402


class FakeClient:
    def __init__(self, blocks=(), texts=None, fail=False):
        self.blocks = [SimpleNamespace(type=t, bbox=b) for t, b in blocks]
        self.texts = texts
        self.fail = fail
        self.calls = []

    def layout_detect(self, image):
        return self.blocks

    def batch_content_extract(self, images, types):
        self.calls.append((len(images), list(types)))
        if self.fail:
            raise RuntimeError("boom")
        return self.texts or [f"{t}-{i}" for i, t in enumerate(types)]


@pytest.fixture
def fake(monkeypatch):
    holder = {}

    def install(client):
        holder["client"] = client
        monkeypatch.setattr(_mineru, "get_client", lambda *a, **k: client)
        return client

    return install


def _region(label, text=""):
    return Region(bbox=BBox(0, 0, 10, 10), image=Image.new("RGB", (10, 10)),
                  label=label, text=text)


def test_detector_scales_boxes_keeps_order_and_marks_ordered(fake):
    from newspaper_ocr.detectors.mineru import MineruDetector

    fake(FakeClient(blocks=[("title", [0.0, 0.0, 1.0, 0.1]),
                            ("text", [0.5, 0.2, 1.0, 0.9]),
                            ("text", [0.0, 0.2, 0.5, 0.9]),
                            ("text", [0.3, 0.3, 0.3, 0.4])]))  # degenerate
    layout = MineruDetector(model_dir=None, skip_lines=True).detect(
        Image.new("RGB", (200, 100), "white"))
    assert layout.ordered and not layout.lines_detected
    assert [r.bbox.to_tuple() for r in layout.regions] == [
        (0, 0, 200, 10), (100, 20, 200, 90), (0, 20, 100, 90)]
    assert [r.label for r in layout.regions] == ["title", "text", "text"]
    assert all(r.confidence == 1.0 for r in layout.regions)
    assert layout.regions[1].image.size == (100, 70)


def test_recognizer_batches_maps_types_and_skips_pictures(fake):
    from newspaper_ocr.recognizers.mineru import MineruRecognizer

    client = fake(FakeClient())
    regions = [_region("title"), _region("image", text="stale"),
               _region("figure"), _region("table")]
    out = MineruRecognizer().recognize_regions(None, regions)
    # One call for the page; the MinerU picture block is skipped, a foreign
    # "figure" label is read as text, MinerU types keep their own prompt.
    assert client.calls == [(3, ["title", "text", "table"])]
    assert [r.text for r in out] == ["title-0", "", "text-1", "table-2"]
    assert all(r.status == "ok" for r in out)


def test_recognizer_marks_errors_and_repetition(fake):
    from newspaper_ocr.recognizers.mineru import MineruRecognizer

    fake(FakeClient(fail=True))
    r = MineruRecognizer().recognize(_region("text"))
    assert (r.text, r.status) == ("", "error")

    loop = "the same phrase again " * 10
    fake(FakeClient(texts=[loop]))
    r = MineruRecognizer().recognize(_region("text"))
    assert r.status == "repetition" and len(r.text) < len(loop)
