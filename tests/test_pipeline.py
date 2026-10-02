from newspaper_ocr.pipeline import Pipeline
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.recognizers.base import LineRecognizer, RegionRecognizer
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
        output=MockFormatter(),
        layout_processing=False,
    )
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    result = pipe.run(img)
    assert "mock text" in result


def test_load_image_accepts_path(tmp_path):
    p = tmp_path / "page.png"
    Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(str(p))
    for arg in (str(p), p):
        img = Pipeline._load_image(arg)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"


def test_load_image_converts_grayscale():
    gray = Image.new("L", (30, 20))
    assert Pipeline._load_image(gray).mode == "RGB"


def test_load_image_rejects_non_image():
    import pytest

    with pytest.raises(TypeError):
        Pipeline._load_image(42)


def test_analyze_accepts_grayscale_and_path(tmp_path):
    # A grayscale page must not crash the detector path (regression: passing a
    # non-RGB image or a path used to fail deep inside detect()).
    p = tmp_path / "gray.png"
    Image.new("L", (200, 100)).save(str(p))
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=MockRecognizer(),
        output=MockFormatter(),
        layout_processing=False,
    )
    assert "mock text" in pipe.run(str(p))
    assert "mock text" in pipe.run(Image.new("L", (200, 100)))


def test_pipeline_ocr_from_path(tmp_path):
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    img_path = tmp_path / "test.png"
    img.save(str(img_path))
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=MockRecognizer(),
        output=MockFormatter(),
        layout_processing=False,
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
        output=MockFormatter(),
        layout_processing=False,
    )
    results = pipe.ocr_batch(paths)
    assert len(results) == 3
    assert all("mock text" in r for r in results)


# ---------------------------------------------------------------------------
# Fallback recognizer tests
# ---------------------------------------------------------------------------

class LowConfRecognizer(LineRecognizer):
    """Primary recognizer that always returns low-confidence results."""
    def recognize(self, line):
        line.text = "low conf"
        line.confidence = 0.3  # 30 on 0-100 scale — below default threshold of 70
        return line


class HighConfRecognizer(LineRecognizer):
    """Primary recognizer that always returns high-confidence results."""
    def recognize(self, line):
        line.text = "high conf"
        line.confidence = 0.95  # 95 on 0-100 scale — above threshold
        return line


class FallbackLineRecognizer(LineRecognizer):
    """Fallback recognizer (LineRecognizer variant)."""
    def __init__(self):
        self.call_count = 0

    def recognize(self, line):
        self.call_count += 1
        line.text = "fallback text"
        line.confidence = 0.95
        return line


class FallbackRegionRecognizer(RegionRecognizer):
    """Fallback recognizer (RegionRecognizer variant)."""
    def __init__(self):
        self.call_count = 0

    def recognize(self, region):
        self.call_count += 1
        region.text = "region fallback"
        return region


def _make_img():
    return Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))


def test_fallback_triggered_when_confidence_low():
    """Fallback LineRecognizer is invoked when primary confidence is below threshold."""
    fallback = FallbackLineRecognizer()
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=LowConfRecognizer(),
        fallback=fallback,
        fallback_threshold=70,
        output=MockFormatter(),
        layout_processing=False,
        text_cleaning=False,
    )
    result = pipe.run(_make_img())
    assert "fallback text" in result
    assert fallback.call_count == 1


def test_fallback_not_triggered_when_confidence_high():
    """Fallback is NOT invoked when primary confidence exceeds threshold."""
    fallback = FallbackLineRecognizer()
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=HighConfRecognizer(),
        fallback=fallback,
        fallback_threshold=70,
        output=MockFormatter(),
        layout_processing=False,
        text_cleaning=False,
    )
    result = pipe.run(_make_img())
    assert "high conf" in result
    assert fallback.call_count == 0


def test_fallback_none_does_not_change_behavior():
    """Pipeline without fallback works exactly as before."""
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=LowConfRecognizer(),
        fallback=None,
        output=MockFormatter(),
        layout_processing=False,
        text_cleaning=False,
    )
    result = pipe.run(_make_img())
    assert "low conf" in result


def test_fallback_region_recognizer_wraps_line():
    """A RegionRecognizer used as fallback is wrapped correctly around a line."""
    fallback = FallbackRegionRecognizer()
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=LowConfRecognizer(),
        fallback=fallback,
        fallback_threshold=70,
        output=MockFormatter(),
        layout_processing=False,
        text_cleaning=False,
    )
    result = pipe.run(_make_img())
    assert "region fallback" in result
    assert fallback.call_count == 1


def test_fallback_threshold_boundary():
    """Line with confidence exactly at threshold is NOT sent to fallback."""
    fallback = FallbackLineRecognizer()

    class ExactThresholdRecognizer(LineRecognizer):
        def recognize(self, line):
            line.text = "exact"
            line.confidence = 0.70  # exactly 70 on 0-100 scale
            return line

    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=ExactThresholdRecognizer(),
        fallback=fallback,
        fallback_threshold=70,
        output=MockFormatter(),
        layout_processing=False,
        text_cleaning=False,
    )
    result = pipe.run(_make_img())
    assert "exact" in result
    assert fallback.call_count == 0


class MarkupRecognizer(RegionRecognizer):
    def recognize(self, region):
        region.text = "<table><tr><td>A</td><td>1</td></tr></table>"
        return region


def _markup_pipeline(**kwargs):
    return Pipeline(
        detector=MockDetector(), recognizer=MarkupRecognizer(),
        output=MockFormatter(), layout_processing=False, residual_ocr=False, **kwargs,
    )


def test_markup_stripped_by_default():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    assert _markup_pipeline().run(img) == "A\t1"


def test_markup_raw_keeps_model_output():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    assert "<td>" in _markup_pipeline(markup="raw").run(img)


def test_markup_rejects_unknown_mode():
    import pytest
    with pytest.raises(ValueError):
        _markup_pipeline(markup="html")


class OverlapDetector(Detector):
    """A page whose layout reads the same paragraph twice (outer + inner box)."""

    def detect(self, image):
        w, h = image.size
        boxes = [BBox(0, 0, w, h), BBox(5, 5, w - 5, h // 2)]
        regions = [Region(bbox=b, image=image.crop(b.to_tuple()), label="text") for b in boxes]
        return PageLayout(image=image, regions=regions, width=w, height=h, ordered=True)


class SameTextRecognizer(RegionRecognizer):
    def recognize(self, region):
        region.text = "Duff withdraws from the race for student body president."
        return region


def _dedup_pipeline(**kwargs):
    return Pipeline(detector=OverlapDetector(), recognizer=SameTextRecognizer(),
                    output=MockFormatter(), residual_ocr=False,
                    layout_processing=False, **kwargs)


def test_region_dedup_on_by_default_for_region_recognizers():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    assert len(_dedup_pipeline().analyze(img).regions) == 1


def test_region_dedup_can_be_turned_off():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    assert len(_dedup_pipeline(region_dedup=False).analyze(img).regions) == 2


# --- empty-read rescue -------------------------------------------------------


class ClassifiedsDetector(Detector):
    """One page-sized primary region (a classifieds page boxed as a 'table'),
    plus a second detector's paragraph boxes inside it as alternates."""

    def __init__(self, label="table"):
        self.label = label

    def detect(self, image):
        w, h = image.size
        big = Region(bbox=BBox(0, 0, w, h), image=image, label=self.label)
        alts = [Region(bbox=BBox(10, 10 + 40 * k, w // 2, 40 + 40 * k), image=None,
                       label="plain_text") for k in range(3)]
        alts.append(Region(bbox=BBox(w // 2, 10, w - 10, 60), image=None, label="figure"))
        return PageLayout(image=image, regions=[big], width=w, height=h,
                          ordered=True, alternates=alts)


class EmptyBigReadRecognizer(RegionRecognizer):
    """Reads the page-sized region as empty and each small box as an ad."""

    def recognize(self, region):
        region.text = "" if region.bbox.x0 == 0 and region.bbox.y0 == 0 else \
            f"FOR RENT: room near campus, ad {region.bbox.y0}"
        return region


def _rescue_pipeline(label="table", **kwargs):
    return Pipeline(detector=ClassifiedsDetector(label), recognizer=EmptyBigReadRecognizer(),
                    output=MockFormatter(), residual_ocr=False, layout_processing=False,
                    **kwargs)


def _page_img():
    return Image.fromarray(np.full((200, 300, 3), 255, dtype=np.uint8))


def test_empty_big_read_is_rescued_from_alternates():
    regions = _rescue_pipeline().analyze(_page_img()).regions
    assert [r.source for r in regions] == ["rescue"] * 3          # figure alternate skipped
    assert all(r.text.startswith("FOR RENT") for r in regions)
    assert [r.bbox.y0 for r in regions] == [10, 50, 90]            # reading order kept


def test_headline_regions_are_not_rescued():
    regions = _rescue_pipeline(label="title").analyze(_page_img()).regions
    assert len(regions) == 1 and regions[0].label == "title"


def test_rescue_can_be_turned_off():
    regions = _rescue_pipeline(rescue_empty_reads=False).analyze(_page_img()).regions
    assert len(regions) == 1 and regions[0].text == ""


def test_rescue_keeps_region_when_alternates_read_less():
    class AllEmpty(RegionRecognizer):
        def recognize(self, region):
            region.text = ""
            return region
    pipe = Pipeline(detector=ClassifiedsDetector(), recognizer=AllEmpty(), output=MockFormatter(),
                    residual_ocr=False, layout_processing=False)
    regions = pipe.analyze(_page_img()).regions
    assert len(regions) == 1 and regions[0].label == "table"


def test_empty_html_table_counts_as_empty():
    class EmptyTable(EmptyBigReadRecognizer):
        def recognize(self, region):
            region = super().recognize(region)
            if region.text == "":
                region.text = "<table>" + "<tr><td></td><td></td></tr>" * 10 + "</table>"
            return region
    pipe = Pipeline(detector=ClassifiedsDetector(), recognizer=EmptyTable(), output=MockFormatter(),
                    residual_ocr=False, layout_processing=False)
    regions = pipe.analyze(_page_img()).regions
    assert [r.source for r in regions] == ["rescue"] * 3


# --- CJK filter --------------------------------------------------------------


class CjkTailRecognizer(RegionRecognizer):
    """Reads the top half as English and the bottom half as invented Chinese."""

    def recognize(self, region):
        region.text = "信" if region.bbox.y0 else "The council met on Tuesday evening."
        return region


class TwoBoxDetector(Detector):
    def detect(self, image):
        w, h = image.size
        regions = [Region(bbox=BBox(0, 0, w, h // 2), image=image, label="text"),
                   Region(bbox=BBox(0, h // 2, w, h), image=image, label="text")]
        return PageLayout(image=image, regions=regions, width=w, height=h)


def _cjk_pipeline(**kwargs):
    return Pipeline(detector=TwoBoxDetector(), recognizer=CjkTailRecognizer(),
                    output=MockFormatter(), residual_ocr=False, layout_processing=False,
                    **kwargs)


def test_cjk_hallucination_is_blanked_by_default():
    regions = _cjk_pipeline().analyze(_page_img()).regions
    assert regions[1].text == "" and regions[1].status == "hallucination"


def test_cjk_filter_can_be_turned_off():
    regions = _cjk_pipeline(cjk_filter=False).analyze(_page_img()).regions
    assert regions[1].text == "信"


# --- reading pictures --------------------------------------------------------


class PictureDetector(Detector):
    """A page-wide picture with an already-boxed text block inside it."""

    def detect(self, image):
        w, h = image.size
        pic = Region(bbox=BBox(0, 0, w, h), image=image, label="image")
        inner = Region(bbox=BBox(10, 10, 60, 40), image=image.crop((10, 10, 60, 40)),
                       label="text")
        return PageLayout(image=image, regions=[pic, inner], width=w, height=h, ordered=True)


class PictureReader(RegionRecognizer):
    """Skips pictures like MinerU; reads any 'text' crop as *answer*."""

    picture_reads = True

    def __init__(self, answer="KEMP'S BALSAM THE BEST COUGH CURE"):
        self.answer = answer
        self.crops = []

    def recognize(self, region):
        if region.label == "image":
            region.text = ""
        elif region.bbox.x0 == 0:          # the picture, re-read as text
            self.crops.append(region.image)
            region.text = self.answer
        else:
            region.text = "Already read."
        return region


def _picture_pipeline(rec, **kwargs):
    return Pipeline(detector=PictureDetector(), recognizer=rec, output=MockFormatter(),
                    residual_ocr=False, layout_processing=False, **kwargs)


def _ink_page():
    return Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))


def test_picture_text_is_read_and_label_kept():
    rec = PictureReader()
    regions = _picture_pipeline(rec).analyze(_ink_page()).regions
    assert regions[0].label == "image"
    assert regions[0].text == "KEMP'S BALSAM THE BEST COUGH CURE"
    # The text box inside the picture was whited out before the re-read.
    crop = np.asarray(rec.crops[0])
    assert crop[20:30, 20:50].min() == 255 and crop[60:90, 100:190].max() == 0


def test_picture_reads_need_three_words_and_no_latex():
    for answer in ("B-A-T", r"\( \frac{1 + u}{7} = 70\% \)"):
        regions = _picture_pipeline(PictureReader(answer)).analyze(_ink_page()).regions
        assert regions[0].text == ""


def test_read_pictures_follows_the_recognizer_unless_forced():
    class Plain(PictureReader):
        picture_reads = False
    assert _picture_pipeline(Plain()).analyze(_ink_page()).regions[0].text == ""
    forced = _picture_pipeline(Plain(), read_pictures=True).analyze(_ink_page())
    assert forced.regions[0].text.startswith("KEMP'S")
    off = _picture_pipeline(PictureReader(), read_pictures=False).analyze(_ink_page())
    assert off.regions[0].text == ""
