"""Tests for the OpenSeadragon review site (issue #1, item 5)."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest
from PIL import Image

from newspaper_ocr.formatters.viewer import ViewerFormatter, status_counts
from newspaper_ocr.models import BBox, PageLayout, Region
from newspaper_ocr.viewer import ReviewSite, build_site


def _img(w: int = 400, h: int = 600) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def _region(y0: int, y1: int, text: str = "", status: str = "ok", **kw) -> Region:
    return Region(
        bbox=BBox(0, y0, 400, y1),
        image=_img(400, y1 - y0),
        label=kw.pop("label", "text"),
        text=text,
        status=status,
        **kw,
    )


def _layout(regions: list[Region]) -> PageLayout:
    return PageLayout(image=_img(), regions=regions, width=400, height=600)


def _page_json(html: str) -> dict:
    """Pull the PAGE object back out of the rendered HTML."""
    match = re.search(r"const PAGE = (\{.*?\});\n", html, re.DOTALL)
    assert match, "PAGE payload not found"
    return json.loads(match.group(1).replace("<\\/", "</"))


class TestViewerFormatter:
    def test_renders_a_region_per_article_and_overlay(self):
        layout = _layout([_region(0, 100, "First"), _region(100, 200, "Second")])
        html = ViewerFormatter().format(layout)

        assert html.count("<article") == 2
        assert _page_json(html)["regions"][0]["bbox"] == [0, 0, 400, 100]

    def test_uses_region_ids_and_falls_back_to_position(self):
        layout = _layout([_region(0, 100, "a", id="page7-r0"), _region(100, 200, "b")])
        page = _page_json(ViewerFormatter().format(layout))
        assert [r["id"] for r in page["regions"]] == ["page7-r0", "r1"]

    def test_status_drives_the_overlay_colour(self):
        layout = _layout(
            [_region(0, 100, "ok text"), _region(100, 200, "[OCR timeout]", "timeout")]
        )
        colors = [r["color"] for r in _page_json(ViewerFormatter().format(layout))["regions"]]
        assert colors[0] != colors[1]

    def test_escapes_region_text(self):
        layout = _layout([_region(0, 100, '<script>alert("xss")</script>')])
        html = ViewerFormatter().format(layout)
        assert "<script>alert" not in html
        assert "&lt;script&gt;alert" in html

    def test_label_cannot_break_out_of_the_script_block(self):
        """A "</script>" inside the JSON payload would end the block early."""
        layout = _layout([_region(0, 100, "x", label="</script><img onerror=1>")])
        html = ViewerFormatter().format(layout)
        assert "</script><img" not in html
        assert _page_json(html)["regions"][0]["label"] == "</script><img onerror=1>"

    def test_empty_region_is_marked_rather_than_blank(self):
        html = ViewerFormatter().format(_layout([_region(0, 100, "")]))
        assert "(no text)" in html

    def test_page_with_no_regions_still_renders(self):
        html = ViewerFormatter().format(_layout([]))
        assert "<article" not in html
        assert _page_json(html)["regions"] == []

    def test_falls_back_to_image_size_when_layout_dims_are_unset(self):
        layout = PageLayout(image=_img(400, 600), regions=[_region(0, 100)])
        page = _page_json(ViewerFormatter().format(layout))
        assert (page["width"], page["height"]) == (400, 600)

    def test_nav_links_are_rendered(self):
        html = ViewerFormatter(nav=[("Index", "index.html")]).format(_layout([]))
        assert '<a href="index.html">Index</a>' in html

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("osd.js", "images/"),
            ("vendor/osd.js", "vendor/images/"),
            ("/static/openseadragon.min.js", "/static/images/"),
            ("https://cdn.example/x/4.1.0/osd.js", "https://cdn.example/x/4.1.0/images/"),
        ],
    )
    def test_sprite_directory_is_derived_from_the_script_url(self, url, expected):
        formatter = ViewerFormatter(openseadragon_url=url)
        assert formatter._images_url() == expected
        assert f'prefixUrl: "{expected}"' in formatter.format(_layout([]))


class TestStatusCounts:
    def test_counts_by_status(self):
        layout = _layout(
            [
                _region(0, 100, "a"),
                _region(100, 200, "b"),
                _region(200, 300, "c", "timeout"),
            ]
        )
        assert status_counts(layout) == {"ok": 2, "timeout": 1}

    def test_omits_statuses_with_no_regions(self):
        assert status_counts(_layout([_region(0, 100)])) == {"ok": 1}


class TestReviewSite:
    def test_writes_pages_index_manifest_and_scans(self, tmp_path):
        site = ReviewSite(tmp_path / "site", title="Test Issue")
        site.add_page(_layout([_region(0, 100, "one")]))
        site.add_page(_layout([_region(0, 100, "two", "timeout")]))
        index = site.write()

        out = tmp_path / "site"
        assert index == out / "index.html"
        assert (out / "page-001.html").is_file()
        assert (out / "page-002.html").is_file()
        assert (out / "manifest.json").is_file()
        assert (out / "scans" / "page-001.jpg").is_file()

    def test_scans_avoid_the_openseadragon_sprite_directory(self, tmp_path):
        """A vendored copy of the library owns images/; scans must not."""
        site = ReviewSite(tmp_path / "site")
        name = site.add_page(_layout([_region(0, 100, "x")]))
        site.write()
        assert not (tmp_path / "site" / "images").exists()
        assert f"scans/{name}.jpg" in (tmp_path / "site" / "page-001.html").read_text()

    def test_manifest_carries_regions_and_statuses(self, tmp_path):
        site = ReviewSite(tmp_path / "site", title="Test Issue")
        site.add_page(
            _layout([_region(0, 100, "kept"), _region(100, 200, "[OCR timeout]", "timeout")])
        )
        site.write()

        manifest = json.loads((tmp_path / "site" / "manifest.json").read_text())
        assert manifest["title"] == "Test Issue"
        assert manifest["page_count"] == 1
        assert manifest["region_count"] == 2
        assert manifest["status_counts"] == {"ok": 1, "timeout": 1}

        region = manifest["pages"][0]["regions"][1]
        assert region["status"] == "timeout"
        assert region["bbox"] == [0, 100, 400, 200]
        assert region["text"] == "[OCR timeout]"

    def test_navigation_links_span_the_issue(self, tmp_path):
        site = ReviewSite(tmp_path / "site")
        for _ in range(3):
            site.add_page(_layout([_region(0, 100, "x")]))
        site.write()

        first = (tmp_path / "site" / "page-001.html").read_text()
        middle = (tmp_path / "site" / "page-002.html").read_text()
        last = (tmp_path / "site" / "page-003.html").read_text()

        assert "page-002.html" in first and "Prev" not in first
        assert "page-001.html" in middle and "page-003.html" in middle
        assert "page-002.html" in last and "Next" not in last

    def test_index_lists_every_page_and_flags_problems(self, tmp_path):
        site = ReviewSite(tmp_path / "site")
        site.add_page(_layout([_region(0, 100, "fine")]))
        site.add_page(_layout([_region(0, 100, "bad", "timeout")]))
        site.write()

        index = (tmp_path / "site" / "index.html").read_text()
        assert index.count('class="card"') == 2
        assert "1 need review" in index

    def test_custom_page_names(self, tmp_path):
        site = ReviewSite(tmp_path / "site")
        assert site.add_page(_layout([]), name="seq-0004") == "seq-0004"
        site.write()
        assert (tmp_path / "site" / "seq-0004.html").is_file()

    def test_png_output(self, tmp_path):
        site = ReviewSite(tmp_path / "site", image_format="PNG")
        site.add_page(_layout([_region(0, 100, "x")]))
        site.write()
        assert (tmp_path / "site" / "scans" / "page-001.png").is_file()

    def test_scan_is_written_when_the_page_is_added(self, tmp_path):
        """Long runs shouldn't hold every decoded page until write()."""
        site = ReviewSite(tmp_path / "site")
        site.add_page(_layout([_region(0, 100, "x")]))
        assert (tmp_path / "site" / "scans" / "page-001.jpg").is_file()

    def test_write_with_no_pages_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="No pages"):
            ReviewSite(tmp_path / "site").write()

    def test_openseadragon_url_reaches_the_pages(self, tmp_path):
        site = ReviewSite(tmp_path / "site", openseadragon_url="osd.js")
        site.add_page(_layout([_region(0, 100, "x")]))
        site.write()
        assert '<script src="osd.js">' in (tmp_path / "site" / "page-001.html").read_text()


class TestBuildSite:
    def test_builds_from_an_iterable_of_layouts(self, tmp_path):
        layouts = (_layout([_region(0, 100, f"page {i}")]) for i in range(3))
        index = build_site(layouts, tmp_path / "site", title="Issue")

        assert index.is_file()
        manifest = json.loads((tmp_path / "site" / "manifest.json").read_text())
        assert manifest["page_count"] == 3


class TestRegistry:
    @pytest.mark.parametrize("name", ["viewer", "html"])
    def test_registered_under_both_names(self, name):
        from newspaper_ocr.formatters import FORMATTERS

        assert FORMATTERS.get(name) is ViewerFormatter
