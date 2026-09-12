"""Build a review website for an OCR'd issue.

Writes what a human needs to check a run: one OpenSeadragon page per scan, an
index over the issue, and a ``manifest.json`` describing every page and region
for whatever comes next (article segmentation, a re-OCR pass, an enrichment
layer).  The manifest is the machine-readable half and carries the same region
fields as the JSON formatter, ``status`` included, so a recovery pass can find
the regions worth redoing without opening the images.

    from newspaper_ocr import Pipeline
    from newspaper_ocr.pdf import page_images
    from newspaper_ocr.viewer import ReviewSite

    pipe = Pipeline(recognizer="glm-ocr")
    site = ReviewSite("site/industrial-worker-1912-05-01", title="Industrial Worker")

    for image in page_images("issue.pdf", rotate=90):
        site.add_page(pipe.analyze(image))

    site.write()
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from newspaper_ocr.formatters.viewer import (
    OPENSEADRAGON_CDN,
    _STATUS_COLORS,
    ViewerFormatter,
    status_counts,
)
from newspaper_ocr.models import PageLayout

_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #ffffff; --fg: #1b1b1b; --muted: #5c5c5c; --rule: #d9d9d9; --panel: #f7f7f5;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #16181c; --fg: #e8e8e6; --muted: #9a9a97; --rule: #2f333a; --panel: #1d2026; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0 auto; max-width: 1100px; padding: 24px 16px 64px;
    background: var(--bg); color: var(--fg);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); margin: 0 0 24px; }}
  .grid {{
    display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  }}
  a.card {{
    display: block; padding: 12px 14px; border: 1px solid var(--rule); border-radius: 8px;
    text-decoration: none; color: inherit; background: var(--panel);
  }}
  a.card:hover {{ border-color: var(--fg); }}
  .n {{ font-weight: 600; margin-bottom: 6px; }}
  .dims {{ color: var(--muted); font-size: 13px; }}
  .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .tag {{
    font: 11px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
    padding: 0 7px; border-radius: 99px; color: #fff;
  }}
  footer {{ margin-top: 32px; color: var(--muted); font-size: 13px; }}
  footer a {{ color: inherit; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="sub">{summary}</p>
<div class="grid">{cards}</div>
<footer>Region data for every page: <a href="manifest.json">manifest.json</a></footer>
</body>
</html>
"""


@dataclass
class _Page:
    name: str
    layout: PageLayout
    image_name: str


class ReviewSite:
    """Collect OCR'd pages and write a static review site.

    Parameters
    ----------
    out_dir:
        Directory to write into.  Created if absent.
    title:
        Heading for the index, e.g. the issue's masthead and date.
    image_format / image_quality:
        How page scans are saved.  JPEG keeps an issue's worth of broadsheets to
        a sane size; pass ``"PNG"`` when the review has to be lossless.
    openseadragon_url:
        Where the pages load OpenSeadragon from.  Defaults to a CDN, so the site
        needs network access; point it at a copy inside ``out_dir`` for a site
        that has to work offline.
    """

    def __init__(
        self,
        out_dir: str | Path,
        title: str = "OCR review",
        image_format: str = "JPEG",
        image_quality: int = 85,
        openseadragon_url: str = OPENSEADRAGON_CDN,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.title = title
        self.image_format = image_format
        self.image_quality = image_quality
        self.openseadragon_url = openseadragon_url
        self._pages: list[_Page] = []

    def add_page(self, layout: PageLayout, name: str | None = None) -> str:
        """Add a page. Returns the name used for its files.

        The scan is written immediately so a long run doesn't accumulate
        decoded images in memory; only the regions are kept until
        :meth:`write`.
        """
        name = name or f"page-{len(self._pages) + 1:03d}"
        suffix = "png" if self.image_format.upper() == "PNG" else "jpg"
        # "scans/", not "images/": OpenSeadragon looks for its control sprites in
        # an images/ directory beside the script, so a vendored copy of the
        # library inside out_dir would collide with the page scans.
        image_name = f"scans/{name}.{suffix}"

        target = self.out_dir / image_name
        target.parent.mkdir(parents=True, exist_ok=True)
        image = layout.image
        if self.image_format.upper() == "JPEG":
            image.convert("RGB").save(target, "JPEG", quality=self.image_quality)
        else:
            image.save(target, self.image_format)

        self._pages.append(_Page(name=name, layout=layout, image_name=image_name))
        return name

    def write(self) -> Path:
        """Write every page's HTML, the index and the manifest. Returns the index path."""
        if not self._pages:
            raise ValueError("No pages added — nothing to write.")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        for i, page in enumerate(self._pages):
            nav = [("Index", "index.html")]
            if i > 0:
                nav.insert(0, ("← Prev", f"{self._pages[i - 1].name}.html"))
            if i < len(self._pages) - 1:
                nav.append(("Next →", f"{self._pages[i + 1].name}.html"))

            formatter = ViewerFormatter(
                image_url=page.image_name,
                title=f"{self.title} · {page.name}",
                nav=nav,
                openseadragon_url=self.openseadragon_url,
            )
            (self.out_dir / f"{page.name}.html").write_text(
                formatter.format(page.layout), encoding="utf-8"
            )

        (self.out_dir / "manifest.json").write_text(
            json.dumps(self.manifest(), indent=2), encoding="utf-8"
        )

        index = self.out_dir / "index.html"
        index.write_text(self._index_html(), encoding="utf-8")
        return index

    def manifest(self) -> dict:
        """The site as data: every page, every region, every status."""
        pages = []
        for page in self._pages:
            layout = page.layout
            pages.append(
                {
                    "name": page.name,
                    "html": f"{page.name}.html",
                    "image": page.image_name,
                    "width": layout.width or layout.image.width,
                    "height": layout.height or layout.image.height,
                    "status_counts": status_counts(layout),
                    "regions": [
                        {
                            "id": r.id or f"r{i}",
                            "label": r.label,
                            "bbox": list(r.bbox.to_tuple()),
                            "text": r.text,
                            "status": r.status,
                            "confidence": r.confidence,
                        }
                        for i, r in enumerate(layout.regions)
                    ],
                }
            )

        totals: dict[str, int] = {}
        for page in pages:
            for status, n in page["status_counts"].items():
                totals[status] = totals.get(status, 0) + n

        return {
            "title": self.title,
            "pages": pages,
            "page_count": len(pages),
            "region_count": sum(len(p["regions"]) for p in pages),
            "status_counts": totals,
        }

    def _index_html(self) -> str:
        manifest = self.manifest()
        cards = []
        for page in manifest["pages"]:
            tags = "".join(
                f'<span class="tag" style="background:{_STATUS_COLORS.get(status, "#777")}">'
                f"{n} {html.escape(status)}</span>"
                for status, n in page["status_counts"].items()
            )
            cards.append(
                f'<a class="card" href="{html.escape(page["html"])}">'
                f'<div class="n">{html.escape(page["name"])}</div>'
                f'<div class="dims">{page["width"]} × {page["height"]} · '
                f'{len(page["regions"])} regions</div>'
                f'<div class="tags">{tags}</div></a>'
            )

        flagged = sum(n for s, n in manifest["status_counts"].items() if s != "ok")
        summary = (
            f"{manifest['page_count']} pages · {manifest['region_count']} regions"
        )
        if flagged:
            summary += f" · {flagged} need review"

        return _INDEX_TEMPLATE.format(
            title=html.escape(self.title),
            summary=html.escape(summary),
            cards="".join(cards),
        )


def build_site(
    layouts,
    out_dir: str | Path,
    title: str = "OCR review",
    **kwargs,
) -> Path:
    """Write a review site from an iterable of :class:`PageLayout`.

    Convenience wrapper over :class:`ReviewSite` for the common case of OCRing
    an issue front to back::

        build_site((pipe.analyze(im) for im in page_images("issue.pdf")), "site/")
    """
    site = ReviewSite(out_dir, title=title, **kwargs)
    for layout in layouts:
        site.add_page(layout)
    return site.write()
