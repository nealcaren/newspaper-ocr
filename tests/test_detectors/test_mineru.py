"""Tests for the MinerU2.5 detector and recognizer, with a fake client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("mineru_vl_utils")

from newspaper_ocr import _mineru  # noqa: E402
from newspaper_ocr.models import BBox, Region  # noqa: E402


class FakeClient:
    def __init__(self, blocks=(), texts=None, fail=False, oom_over=None,
                 fail_types=(), layout_exc=None):
        self.blocks = [SimpleNamespace(type=t, bbox=b) for t, b in blocks]
        self.texts = texts
        self.fail = fail
        self.oom_over = oom_over
        self.fail_types = set(fail_types)
        self.layout_exc = layout_exc
        self.calls = []

    def layout_detect(self, image):
        if self.layout_exc:
            raise self.layout_exc
        return self.blocks

    def batch_content_extract(self, images, types):
        self.calls.append((len(images), list(types)))
        if self.fail or self.fail_types & set(types):
            raise RuntimeError("boom")
        if self.oom_over is not None and len(images) > self.oom_over:
            raise RuntimeError("MPS backend out of memory")
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


def test_recognizer_retries_one_by_one_after_batch_failure(fake):
    from newspaper_ocr.recognizers.mineru import MineruRecognizer

    client = fake(FakeClient(oom_over=1, fail_types={"table"}))
    regions = [_region("text"), _region("table"), _region("title")]
    out = MineruRecognizer().recognize_regions(None, regions)
    assert client.calls[0] == (3, ["text", "table", "title"])
    assert [len(types) for _, types in client.calls[1:]] == [1, 1, 1]
    assert [r.status for r in out] == ["ok", "error", "ok"]
    assert out[0].text and out[2].text and out[1].text == ""


def test_detector_turns_oom_into_actionable_memory_error(fake):
    from newspaper_ocr.detectors.mineru import MineruDetector

    fake(FakeClient(layout_exc=RuntimeError("MPS backend out of memory (MPS allocated: 8 GB)")))
    with pytest.raises(MemoryError, match="NEWSPAPER_OCR_MPS_MEMORY_FRACTION"):
        MineruDetector().detect(Image.new("RGB", (50, 50)))

    fake(FakeClient(layout_exc=ValueError("bad output")))
    with pytest.raises(ValueError):
        MineruDetector().detect(Image.new("RGB", (50, 50)))


# --- backend selection ------------------------------------------------------


@pytest.fixture
def recorded_clients(monkeypatch):
    """Record MinerUClient construction instead of loading a model."""
    made = []

    class RecordingClient:
        def __init__(self, **kwargs):
            made.append(kwargs)

    import mineru_vl_utils
    monkeypatch.setattr(mineru_vl_utils, "MinerUClient", RecordingClient)
    monkeypatch.setattr(_mineru, "_vllm_engine", lambda model: f"engine:{model}")
    monkeypatch.setattr(_mineru, "_CLIENTS", {})
    monkeypatch.delenv("MINERU_SERVER_URL", raising=False)
    return made


def test_vllm_backend_shares_one_engine_between_detector_and_recognizer(recorded_clients):
    from newspaper_ocr.detectors import DETECTORS
    from newspaper_ocr.recognizers import RECOGNIZERS

    det = DETECTORS.get("mineru-vllm")(model_dir=None, skip_lines=False)
    rec = RECOGNIZERS.get("mineru-vllm")()
    assert det.client is rec.client
    assert len(recorded_clients) == 1
    assert recorded_clients[0]["backend"] == "vllm-engine"
    assert recorded_clients[0]["vllm_llm"] == f"engine:{_mineru.DEFAULT_MODEL}"
    assert det.device == "cuda"


def test_http_backend_reads_server_url_from_env(recorded_clients, monkeypatch):
    monkeypatch.setenv("MINERU_SERVER_URL", "http://gpu01:30000")
    from newspaper_ocr.recognizers import RECOGNIZERS

    rec = RECOGNIZERS.get("mineru-http")()
    assert recorded_clients[0]["backend"] == "http-client"
    assert recorded_clients[0]["server_url"] == "http://gpu01:30000"
    assert rec.device == "remote"


def test_http_backend_without_server_is_actionable(recorded_clients):
    with pytest.raises(ValueError, match="MINERU_SERVER_URL"):
        _mineru.get_client(backend="http")


def test_unknown_backend_rejected(recorded_clients):
    with pytest.raises(ValueError, match="backend"):
        _mineru.get_client(backend="sglang")


class SplitClient(FakeClient):
    """Returns *picture_calls* layouts as given (one per call, in order), then
    boxes every later tile as two stacked text blocks."""

    def __init__(self, *picture_calls):
        super().__init__()
        self.picture_calls = [[SimpleNamespace(type=t, bbox=b) for t, b in blocks]
                              for blocks in picture_calls]
        self.sizes = []

    def layout_detect(self, image):
        self.sizes.append(image.size)
        if len(self.sizes) <= len(self.picture_calls):
            return self.picture_calls[len(self.sizes) - 1]
        return [SimpleNamespace(type="text", bbox=[0.1, 0.0, 0.9, 0.5]),
                SimpleNamespace(type="text", bbox=[0.1, 0.5, 0.9, 1.0])]


PAGE_PICTURE = [("image", [0.0, 0.0, 1.0, 1.0])]


def test_page_sized_picture_is_retried_in_halves(fake):
    from newspaper_ocr.detectors.mineru import MineruDetector

    client = fake(SplitClient(PAGE_PICTURE))
    det = MineruDetector()
    layout = det.detect(Image.new("RGB", (200, 100), "white"))
    assert det.last_tiles == 2 and client.sizes == [(200, 100), (110, 100), (110, 100)]
    # Left half first, then right; boxes in page coordinates.
    assert [r.bbox.to_tuple() for r in layout.regions] == [
        (11, 0, 99, 50), (11, 50, 99, 100), (101, 0, 189, 50), (101, 50, 189, 100)]


def test_a_half_that_misfires_again_is_split_top_and_bottom(fake):
    from newspaper_ocr.detectors.mineru import MineruDetector

    client = fake(SplitClient(PAGE_PICTURE, PAGE_PICTURE))  # page, then left half
    det = MineruDetector()
    layout = det.detect(Image.new("RGB", (200, 100), "white"))
    assert det.last_tiles == 3
    assert client.sizes == [(200, 100), (110, 100), (110, 55), (110, 55), (110, 100)]
    # Left half top-to-bottom, then the right half.  The left half's middle
    # block straddled the cut and was boxed in both pieces: it becomes one box.
    assert [r.bbox.to_tuple() for r in layout.regions] == [
        (11, 0, 99, 28), (11, 28, 99, 73), (11, 73, 99, 100),
        (101, 0, 189, 50), (101, 50, 189, 100)]
    assert layout.regions[1].image.size == (88, 45)


def test_ordinary_page_is_not_retried(fake):
    from newspaper_ocr.detectors.mineru import MineruDetector

    client = fake(SplitClient([("image", [0.0, 0.0, 0.5, 0.5]),
                               ("text", [0.5, 0.0, 1.0, 1.0])]))
    det = MineruDetector()
    det.detect(Image.new("RGB", (200, 100), "white"))
    assert det.last_tiles == 1 and len(client.sizes) == 1


def test_split_kept_only_if_it_covers_more_text(fake):
    from newspaper_ocr.detectors.mineru import MineruDetector

    blocks = [("text", [0.0, 0.0, 1.0, 1.0])] * MineruDetector.SPLIT_MIN_BLOCKS
    client = fake(SplitClient(blocks))
    det = MineruDetector()
    layout = det.detect(Image.new("RGB", (200, 100), "white"))
    assert len(client.sizes) == 3 and det.last_tiles == 1
    assert len(layout.regions) == MineruDetector.SPLIT_MIN_BLOCKS


def test_split_retry_can_be_turned_off(fake):
    from newspaper_ocr.detectors.mineru import MineruDetector

    client = fake(SplitClient(PAGE_PICTURE))
    MineruDetector(split_retry=False).detect(Image.new("RGB", (200, 100), "white"))
    assert len(client.sizes) == 1
