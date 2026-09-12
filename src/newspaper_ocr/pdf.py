"""PDF front-end: turn a scanned newspaper PDF into per-page images.

Most of our inputs are multi-page scans rather than loose page images.  Each page
usually carries the scan as an embedded image, which is what we want to OCR —
re-rendering it would only resample something that is already a bitmap.  Pages
without one (born-digital, or vector overlays) are rendered instead.

Two wrinkles this handles, both learned from the collection:

*Pick the largest embedded image, not the first.*  Google-scanned PDFs place a
small "Digitized by Google" strip ahead of the page scan, so ``images[0]`` yields
a sliver a few hundred pixels tall instead of the full broadsheet.

*Rotation.*  Many broadsheet scans arrive 90 degrees off, and titles in the same
collection were scanned in opposite directions, so the correction has to be a
per-run choice rather than a constant.  Layout detection on a sideways page
produces nothing usable, so this has to happen before the page reaches the
pipeline.

Requires PyMuPDF: ``pip install "newspaper-ocr[pdf]"``.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

#: Fallback render resolution for pages with no embedded scan.
DEFAULT_DPI = 300

# Clockwise degrees -> PIL transpose. Transpose is lossless and exact, unlike
# rotate(), which resamples; PIL's ROTATE_* constants are counter-clockwise.
_ROTATIONS = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


def _require_pymupdf():
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "PyMuPDF is not installed. Install with:\n"
            '  pip install "newspaper-ocr[pdf]"'
        )
    return pymupdf


def rotate_image(image: Image.Image, degrees: int) -> Image.Image:
    """Rotate *degrees* clockwise. Accepts 0, 90, 180 or 270."""
    if degrees == 0:
        return image
    if degrees not in _ROTATIONS:
        raise ValueError(
            f"rotate must be 0, 90, 180 or 270 (clockwise), got {degrees!r}"
        )
    return image.transpose(_ROTATIONS[degrees])


def _largest_embedded_image(doc, page) -> Image.Image | None:
    """Return the page's largest embedded image by pixel area, if any."""
    best_xref = None
    best_area = 0
    for info in page.get_images(full=True):
        # (xref, smask, width, height, ...) — see PyMuPDF's Page.get_images.
        xref, width, height = info[0], info[2], info[3]
        area = width * height
        if area > best_area:
            best_xref, best_area = xref, area

    if best_xref is None:
        return None

    extracted = doc.extract_image(best_xref)
    if not extracted or not extracted.get("image"):
        return None
    return Image.open(io.BytesIO(extracted["image"])).convert("RGB")


def page_image(
    doc,
    page,
    dpi: int = DEFAULT_DPI,
    rotate: int = 0,
    prefer_embedded: bool = True,
) -> Image.Image:
    """Extract one page as an RGB image, rotated *rotate* degrees clockwise.

    Uses the largest embedded image when there is one, otherwise renders the
    page at *dpi*.  Pass ``prefer_embedded=False`` to always render — useful when
    a page's text is vector rather than scanned.
    """
    image = _largest_embedded_image(doc, page) if prefer_embedded else None
    if image is None:
        pixmap = page.get_pixmap(dpi=dpi)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    return rotate_image(image, rotate)


def page_images(
    path: str | Path,
    dpi: int = DEFAULT_DPI,
    rotate: int = 0,
    pages: range | list[int] | None = None,
    prefer_embedded: bool = True,
) -> Iterator[Image.Image]:
    """Yield one RGB image per page of the PDF at *path*.

    Parameters
    ----------
    dpi:
        Render resolution for pages with no embedded scan.
    rotate:
        Clockwise rotation applied to every page — 0, 90, 180 or 270.
    pages:
        Zero-based page numbers to read.  Defaults to the whole document.
    prefer_embedded:
        Use the largest embedded image when present (the default) rather than
        re-rendering a page that is already a bitmap.

    Pages are yielded lazily, so a long run holds one page in memory at a time
    rather than the whole document.
    """
    pymupdf = _require_pymupdf()
    rotate_image(Image.new("RGB", (1, 1)), rotate)  # validate before opening

    with pymupdf.open(str(path)) as doc:
        numbers = range(doc.page_count) if pages is None else pages
        for number in numbers:
            yield page_image(
                doc,
                doc[number],
                dpi=dpi,
                rotate=rotate,
                prefer_embedded=prefer_embedded,
            )
