"""Tests for the PDF front-end (issue #1, items 4 and 7)."""
from __future__ import annotations

import io

import pytest
from PIL import Image

pymupdf = pytest.importorskip("pymupdf", reason="PDF support is an optional extra")

from newspaper_ocr.pdf import page_images, rotate_image  # noqa: E402


def _png(w: int, h: int, color: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def google_scan(tmp_path):
    """A PDF shaped like a Google scan: banner strip embedded before the page.

    The banner is deliberately first in the page's image list and far smaller
    than the scan, which is exactly the case that makes ``images[0]`` wrong.
    """
    path = tmp_path / "google.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(0, 0, 612, 60), stream=_png(1000, 200, "red"))
    page.insert_image(pymupdf.Rect(0, 60, 612, 792), stream=_png(2400, 3000, "blue"))
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def two_page_pdf(tmp_path):
    path = tmp_path / "issue.pdf"
    doc = pymupdf.open()
    for color in ("blue", "green"):
        page = doc.new_page(width=612, height=792)
        page.insert_image(pymupdf.Rect(0, 0, 612, 792), stream=_png(1200, 1600, color))
    doc.save(str(path))
    doc.close()
    return path


class TestEmbeddedImageSelection:
    def test_picks_largest_embedded_image_not_the_first(self, google_scan):
        (image,) = page_images(google_scan)
        assert image.size == (2400, 3000)
        # Blue is the scan; red would mean we grabbed the banner.
        assert image.getpixel((5, 5)) == (0, 0, 255)

    def test_renders_pages_with_no_embedded_image(self, tmp_path):
        path = tmp_path / "vector.pdf"
        doc = pymupdf.open()
        doc.new_page(width=612, height=792).insert_text((72, 72), "Born digital")
        doc.save(str(path))
        doc.close()

        (image,) = page_images(path, dpi=150)
        # 612pt at 150dpi = 8.5in * 150 = 1275px wide.
        assert image.size == (1275, 1650)

    def test_prefer_embedded_false_forces_a_render(self, google_scan):
        (image,) = page_images(google_scan, dpi=72, prefer_embedded=False)
        assert image.size == (612, 792)


class TestRotation:
    def test_ninety_degrees_is_clockwise(self):
        image = Image.new("RGB", (100, 200), "white")
        image.putpixel((0, 0), (255, 0, 0))

        rotated = rotate_image(image, 90)
        w, _h = rotated.size

        assert rotated.size == (200, 100)
        # Clockwise sends the top-left corner to the top-right.
        assert rotated.getpixel((w - 1, 0)) == (255, 0, 0)

    def test_rotation_is_applied_to_pdf_pages(self, google_scan):
        (image,) = page_images(google_scan, rotate=90)
        assert image.size == (3000, 2400)

    def test_one_eighty_preserves_dimensions(self, google_scan):
        (image,) = page_images(google_scan, rotate=180)
        assert image.size == (2400, 3000)

    def test_zero_is_a_noop(self, google_scan):
        (image,) = page_images(google_scan, rotate=0)
        assert image.size == (2400, 3000)

    @pytest.mark.parametrize("degrees", [45, -90, 360, "90"])
    def test_rejects_other_angles(self, degrees):
        with pytest.raises(ValueError, match="rotate must be"):
            rotate_image(Image.new("RGB", (10, 10)), degrees)

    def test_bad_angle_is_rejected_before_the_file_is_opened(self, tmp_path):
        """Validation shouldn't depend on the PDF being readable."""
        with pytest.raises(ValueError, match="rotate must be"):
            next(page_images(tmp_path / "does-not-exist.pdf", rotate=45))


class TestMultiPage:
    def test_yields_one_image_per_page(self, two_page_pdf):
        images = list(page_images(two_page_pdf))
        assert len(images) == 2
        assert images[0].getpixel((5, 5)) == (0, 0, 255)
        assert images[1].getpixel((5, 5)) == (0, 128, 0)

    def test_pages_selects_a_subset(self, two_page_pdf):
        images = list(page_images(two_page_pdf, pages=[1]))
        assert len(images) == 1
        assert images[0].getpixel((5, 5)) == (0, 128, 0)

    def test_is_lazy(self, two_page_pdf):
        """Pages are produced on demand, not buffered up front."""
        import inspect

        assert inspect.isgenerator(page_images(two_page_pdf))


class TestPipelineIntegration:
    def test_ocr_pdf_runs_the_pipeline_per_page(self, two_page_pdf):
        from newspaper_ocr.pipeline import Pipeline

        pipe = Pipeline.__new__(Pipeline)
        seen: list[tuple[int, int]] = []

        def _run(image):
            seen.append(image.size)
            return f"page {len(seen)}"

        pipe.run = _run

        results = pipe.ocr_pdf(two_page_pdf, rotate=90)

        assert results == ["page 1", "page 2"]
        # Rotation reaches the pipeline, not just the caller.
        assert seen == [(1600, 1200), (1600, 1200)]
