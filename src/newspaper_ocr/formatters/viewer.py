"""OpenSeadragon review page: the scan, its region boxes, and the text beside it.

The point of the review site is checking OCR against the image without squinting
at coordinates: click a box on the scan and the matching text scrolls into view,
click a paragraph and its box lights up on the page.  Regions are tinted by
:attr:`~newspaper_ocr.models.Region.status`, so timeouts and truncated loops are
visible at a glance rather than buried in the JSON.

The page HTML is self-contained apart from OpenSeadragon (pulled from a CDN) and
the scan itself, which it references by URL.  :class:`newspaper_ocr.viewer.ReviewSite`
writes the images and wires the URLs up; using this formatter on its own gives
one page that expects ``page.jpg`` beside it.
"""
from __future__ import annotations

import html
import json

from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.models import PageLayout

OPENSEADRAGON_CDN = (
    "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"
)

#: Tint per region status. "ok" is deliberately the quietest — a clean page
#: should read as clean, with only the problems drawing the eye.
_STATUS_COLORS = {
    "ok": "#2f6fd0",
    "timeout": "#d04a2f",
    "repetition": "#c98a00",
    "error": "#b3179b",
}

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #ffffff;
    --fg: #1b1b1b;
    --muted: #5c5c5c;
    --rule: #d9d9d9;
    --panel: #f7f7f5;
    --sel: #fff3c4;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #16181c;
      --fg: #e8e8e6;
      --muted: #9a9a97;
      --rule: #2f333a;
      --panel: #1d2026;
      --sel: #4a3c00;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  header {{
    display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline;
    padding: 10px 16px; border-bottom: 1px solid var(--rule);
  }}
  header h1 {{ font-size: 16px; margin: 0; font-weight: 600; }}
  header nav {{ margin-left: auto; display: flex; gap: 12px; }}
  header a {{ color: inherit; }}
  .counts {{ color: var(--muted); font-size: 13px; }}
  .counts b {{ font-weight: 600; color: var(--fg); }}
  main {{ display: grid; grid-template-columns: 1fr 420px; height: calc(100vh - 49px); }}
  #osd {{ background: var(--panel); }}
  #pane {{ border-left: 1px solid var(--rule); overflow-y: auto; padding: 8px 0 40vh; }}
  article {{
    padding: 10px 16px; border-bottom: 1px solid var(--rule); cursor: pointer;
  }}
  article:hover {{ background: var(--panel); }}
  article.sel {{ background: var(--sel); }}
  article .meta {{
    font: 11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--muted); text-transform: uppercase; letter-spacing: .04em;
    display: flex; gap: 8px; align-items: center; margin-bottom: 4px;
  }}
  article .dot {{ width: 8px; height: 8px; border-radius: 50%; flex: none; }}
  article p {{ margin: 0; white-space: pre-wrap; }}
  article p.empty {{ color: var(--muted); font-style: italic; }}
  /* Overlays are visual only. OpenSeadragon's mouse tracker swallows events on
     its canvas, so clicks are hit-tested from the viewer instead (see below);
     keeping pointer-events off leaves drag-to-pan working over a box. */
  .box {{
    outline: 2px solid currentColor; background: currentColor; opacity: .16;
    pointer-events: none;
  }}
  .box.sel {{ opacity: .38; outline-width: 3px; }}
  @media (max-width: 800px) {{
    main {{ grid-template-columns: 1fr; height: auto; }}
    #osd {{ height: 60vh; }}
    #pane {{ border-left: 0; border-top: 1px solid var(--rule); padding-bottom: 0; }}
  }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <span class="counts">{counts}</span>
  <nav>{nav}</nav>
</header>
<main>
  <div id="osd"></div>
  <div id="pane">{articles}</div>
</main>
<script src="{cdn}"></script>
<script>
const PAGE = {page_json};
const viewer = OpenSeadragon({{
  id: "osd",
  prefixUrl: "{osd_images}",
  tileSources: {{ type: "image", url: PAGE.image, buildPyramid: false }},
  showNavigator: true,
  // A broadsheet is far taller than the viewer, so fitting one needs to zoom
  // well below 1:1. minZoomImageRatio is what allows that — raise it and the
  // page opens cropped, which is the last thing a review page should do.
  minZoomImageRatio: 0.1,
  visibilityRatio: 0.5,
}});

const articles = new Map();
document.querySelectorAll("article").forEach(el => articles.set(el.dataset.id, el));
let current = null;

function select(id, scroll) {{
  if (current === id) return;
  for (const key of [current, id]) {{
    if (!key) continue;
    const art = articles.get(key);
    const box = document.getElementById("box-" + key);
    if (art) art.classList.toggle("sel", key === id);
    if (box) box.classList.toggle("sel", key === id);
  }}
  current = id;
  if (scroll && articles.has(id)) {{
    articles.get(id).scrollIntoView({{ block: "nearest", behavior: "smooth" }});
  }}
}}

viewer.addHandler("open", () => {{
  // Overlay rects are in viewport coordinates: x and width are fractions of the
  // image width, and y and height are scaled by the same factor, so a tall page
  // extends past y = 1.
  const scale = 1 / PAGE.width;
  for (const r of PAGE.regions) {{
    const el = document.createElement("div");
    el.id = "box-" + r.id;
    el.className = "box";
    el.style.color = r.color;
    viewer.addOverlay({{
      element: el,
      location: new OpenSeadragon.Rect(
        r.bbox[0] * scale,
        r.bbox[1] * scale,
        (r.bbox[2] - r.bbox[0]) * scale,
        (r.bbox[3] - r.bbox[1]) * scale
      ),
    }});
  }}
}});

// Hit-test clicks on the scan against region boxes. event.quick is false for a
// drag, so panning never selects; the smallest region containing the point wins,
// so a box nested inside another is still reachable.
viewer.addHandler("canvas-click", e => {{
  if (!e.quick) return;
  const pt = viewer.viewport.viewportToImageCoordinates(
    viewer.viewport.pointFromPixel(e.position)
  );
  let hit = null;
  let hitArea = Infinity;
  for (const r of PAGE.regions) {{
    const [x0, y0, x1, y1] = r.bbox;
    if (pt.x < x0 || pt.x > x1 || pt.y < y0 || pt.y > y1) continue;
    const area = (x1 - x0) * (y1 - y0);
    if (area < hitArea) {{ hit = r.id; hitArea = area; }}
  }}
  if (hit) select(hit, true);
}});

for (const [id, el] of articles) {{
  el.addEventListener("click", () => {{
    select(id, false);
    const r = PAGE.regions.find(x => x.id === id);
    if (!r) return;
    const scale = 1 / PAGE.width;
    viewer.viewport.fitBounds(new OpenSeadragon.Rect(
      r.bbox[0] * scale,
      r.bbox[1] * scale,
      (r.bbox[2] - r.bbox[0]) * scale,
      (r.bbox[3] - r.bbox[1]) * scale
    ), false);
  }});
}}
</script>
</body>
</html>
"""


def _script_safe_json(data) -> str:
    """JSON that can't terminate the <script> element holding it.

    A detector label or region id containing "</script>" would otherwise close
    the block and drop the rest of the page's JavaScript into the document.
    """
    return json.dumps(data).replace("</", "<\\/")


def status_counts(layout: PageLayout) -> dict[str, int]:
    """Count regions by status, in :data:`_STATUS_COLORS` order, omitting zeroes."""
    counts: dict[str, int] = {}
    for region in layout.regions:
        counts[region.status] = counts.get(region.status, 0) + 1
    return {k: counts[k] for k in _STATUS_COLORS if k in counts} | {
        k: v for k, v in counts.items() if k not in _STATUS_COLORS
    }


class ViewerFormatter(Formatter):
    """Render a page as a standalone OpenSeadragon review page.

    Parameters
    ----------
    image_url:
        URL of the scan, relative to the HTML file.
    title:
        Heading for the page.  Defaults to the image's name.
    nav:
        ``(label, href)`` pairs for the header — previous/next/index links when
        the page is part of a site.
    openseadragon_url:
        Where to load OpenSeadragon from.  Defaults to a CDN, so the page needs
        network access to work; point it at a local copy of
        ``openseadragon.min.js`` for an offline or archival site (its control
        sprites are looked up in an ``images/`` directory beside it).
    """

    def __init__(
        self,
        image_url: str = "page.jpg",
        title: str | None = None,
        nav: list[tuple[str, str]] | None = None,
        openseadragon_url: str = OPENSEADRAGON_CDN,
    ) -> None:
        self.image_url = image_url
        self.title = title
        self.nav = nav or []
        self.openseadragon_url = openseadragon_url

    def _images_url(self) -> str:
        """Where OpenSeadragon looks for its control sprites.

        They live in an ``images/`` directory beside the script, so this is the
        script URL with its filename replaced — including when that URL is a
        bare filename with no directory part at all.
        """
        base, sep, _name = self.openseadragon_url.rpartition("/")
        return f"{base}{sep}images/"

    def format(self, layout: PageLayout) -> str:
        regions = []
        articles = []
        for i, region in enumerate(layout.regions):
            region_id = region.id or f"r{i}"
            color = _STATUS_COLORS.get(region.status, "#777777")
            regions.append(
                {
                    "id": region_id,
                    "label": region.label,
                    "status": region.status,
                    "color": color,
                    "bbox": list(region.bbox.to_tuple()),
                }
            )
            body = (
                f"<p>{html.escape(region.text)}</p>"
                if region.text
                else '<p class="empty">(no text)</p>'
            )
            articles.append(
                f'<article data-id="{html.escape(region_id)}">'
                f'<div class="meta">'
                f'<span class="dot" style="background:{color}"></span>'
                f"<span>{html.escape(region_id)}</span>"
                f"<span>{html.escape(region.label)}</span>"
                f"<span>{html.escape(region.status)}</span>"
                f"</div>{body}</article>"
            )

        counts = status_counts(layout)
        counts_text = " · ".join(
            f"<b>{n}</b> {html.escape(status)}" for status, n in counts.items()
        )
        return _TEMPLATE.format(
            title=html.escape(self.title or self.image_url),
            cdn=html.escape(self.openseadragon_url),
            osd_images=html.escape(self._images_url()),
            counts=counts_text or "<b>0</b> regions",
            nav="".join(
                f'<a href="{html.escape(href)}">{html.escape(label)}</a>'
                for label, href in self.nav
            ),
            articles="".join(articles),
            page_json=_script_safe_json(
                {
                    "image": self.image_url,
                    "width": layout.width or layout.image.width,
                    "height": layout.height or layout.image.height,
                    "regions": regions,
                }
            ),
        )
