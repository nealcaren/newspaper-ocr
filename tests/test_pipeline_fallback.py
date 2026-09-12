"""Region-level chunking and do-no-harm fallback in the Pipeline."""

from PIL import Image

from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.models import BBox, PageLayout, Region
from newspaper_ocr.pipeline import Pipeline
from newspaper_ocr.recognizers.base import RegionRecognizer


class _NoopDetector(Detector):
    def detect(self, image):
        return PageLayout(image=image, regions=[], width=0, height=0)


class _HeightSensitive(RegionRecognizer):
    """Times out on images taller than *tall_px*, otherwise returns band text."""

    def __init__(self, tall_px=500):
        self.tall_px = tall_px
        self.seen_heights = []

    def recognize(self, region):
        h = region.image.size[1]
        self.seen_heights.append(h)
        if h > self.tall_px:
            region.text, region.status = "[OCR timeout]", "timeout"
        else:
            region.text, region.status = f"band{h}", "ok"
        return region


class _Fixed(RegionRecognizer):
    def __init__(self, text, status):
        self.text, self.status = text, status

    def recognize(self, region):
        region.text, region.status = self.text, self.status
        return region


def _pipe(recognizer, fallback=None, **kw):
    return Pipeline(
        detector=_NoopDetector(),
        recognizer=recognizer,
        output="text",
        fallback=fallback,
        **kw,
    )


def _region(h, status="timeout", text="[OCR timeout]"):
    return Region(
        bbox=BBox(0, 0, 300, h),
        image=Image.new("RGB", (300, h), "white"),
        label="plain_text",
        text=text,
        status=status,
    )


class TestChunkRegion:
    def test_all_bands_ok_gives_ok(self):
        pipe = _pipe(_HeightSensitive(tall_px=500), chunk_tall_regions=True,
                     chunk_height=500, chunk_overlap=50)
        out = pipe._chunk_region(_region(1200))
        assert out.status == "ok"
        assert "band" in out.text
        # three bands: 500, 500, 300 -> all <= 500, none time out
        assert pipe.recognizer.seen_heights == [500, 500, 300]

    def test_some_band_timeout_gives_chunked_partial(self):
        # tall_px=400 => the 500px bands time out, the 300px tail is read
        pipe = _pipe(_HeightSensitive(tall_px=400), chunk_tall_regions=True,
                     chunk_height=500, chunk_overlap=50)
        out = pipe._chunk_region(_region(1200))
        assert out.status == "chunked_partial"
        assert out.text == "band300"

    def test_nothing_recovered_keeps_timeout(self):
        pipe = _pipe(_HeightSensitive(tall_px=100), chunk_tall_regions=True,
                     chunk_height=500, chunk_overlap=50)
        out = pipe._chunk_region(_region(1200))
        assert out.status == "timeout"
        assert out.text == "[OCR timeout]"


class TestRegionFallback:
    def test_timeout_replaced_by_any_usable_read(self):
        pipe = _pipe(_Fixed("x", "ok"), fallback=_Fixed("recovered text", "ok"))
        out = pipe._apply_region_fallback(_region(200, "timeout", "[OCR timeout]"))
        assert out.status == "ok"
        assert out.text == "recovered text"
        assert out.text_primary == "[OCR timeout]"
        assert out.engine == "_Fixed"

    def test_no_loss_accepts_repetition(self):
        pipe = _pipe(_Fixed("x", "ok"), fallback=_Fixed("looped bits", "repetition"))
        out = pipe._apply_region_fallback(_region(200, "error", ""))
        assert out.status == "repetition"
        assert out.text == "looped bits"

    def test_partial_not_replaced_by_repetition(self):
        pipe = _pipe(_Fixed("x", "ok"), fallback=_Fixed("still looping", "repetition"))
        region = _region(200, "repetition", "partial real text")
        out = pipe._apply_region_fallback(region)
        # do no harm: partial text stays because the fallback read isn't clean
        assert out.status == "repetition"
        assert out.text == "partial real text"
        assert out.text_primary == ""

    def test_partial_replaced_only_by_clean_read(self):
        pipe = _pipe(_Fixed("x", "ok"), fallback=_Fixed("clean full text", "ok"))
        out = pipe._apply_region_fallback(_region(200, "repetition", "partial"))
        assert out.status == "ok"
        assert out.text == "clean full text"
        assert out.text_primary == "partial"

    def test_ok_region_is_untouched(self):
        fb = _Fixed("should not run", "ok")
        pipe = _pipe(_Fixed("x", "ok"), fallback=fb)
        out = pipe._apply_region_fallback(_region(200, "ok", "good text"))
        assert out.text == "good text"
        assert out.engine == ""

    def test_useless_fallback_read_is_ignored(self):
        pipe = _pipe(_Fixed("x", "ok"), fallback=_Fixed("[OCR timeout]", "timeout"))
        out = pipe._apply_region_fallback(_region(200, "timeout", "[OCR timeout]"))
        assert out.status == "timeout"
        assert out.text_primary == ""


class TestAnalyzeIntegration:
    def test_ladder_chunk_then_fallback(self):
        # primary always times out (whole region and every band), so chunking
        # recovers nothing and the fallback recognizer recovers the region.
        pipe = _pipe(
            _Fixed("[OCR timeout]", "timeout"),
            fallback=_Fixed("fallback recovered", "ok"),
            chunk_tall_regions=True,
            layout_processing=False,
        )
        layout = PageLayout(
            image=Image.new("RGB", (300, 1200)),
            regions=[_region(1200)],
            width=300,
            height=1200,
        )
        # drive just the region-recognition stage via analyze() with a prebuilt layout
        pipe.detector = type("D", (Detector,), {"detect": lambda self, img: layout})()
        out = pipe.analyze(Image.new("RGB", (300, 1200)))
        r = out.regions[0]
        assert r.status == "ok"
        assert r.text == "fallback recovered"
        assert r.engine == "_Fixed"
