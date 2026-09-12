"""Vertical chunking for tall regions.

A region-level VLM can time out on a very tall column. Splitting the crop into
overlapping horizontal bands, OCRing each, and stitching the text back together
recovers those regions. The stitch removes the text duplicated by the pixel
overlap between adjacent bands.

Ported from the production dangerouspress-ocr pipeline (``chunked_ocr`` /
``_deduplicate_chunks``); the defaults match it (500px bands, 50px overlap).
"""
from __future__ import annotations

#: Default band height and overlap, in pixels (production values).
CHUNK_HEIGHT = 500
CHUNK_OVERLAP = 50


def chunk_spans(
    height: int, chunk_height: int = CHUNK_HEIGHT, overlap: int = CHUNK_OVERLAP
) -> list[tuple[int, int]]:
    """Return ``(y0, y1)`` vertical bands covering *height* with *overlap*.

    A region no taller than *chunk_height* yields a single full-height span.
    """
    if height <= chunk_height:
        return [(0, height)]

    spans: list[tuple[int, int]] = []
    y = 0
    while y < height:
        y_end = min(y + chunk_height, height)
        spans.append((y, y_end))
        if y_end == height:
            break
        y = y_end - overlap
    return spans


def merge_chunk_texts(texts: list[str], overlap_chars: int = 80) -> str:
    """Concatenate chunk *texts*, removing the text duplicated across the seam.

    For each adjoining pair, find the longest suffix of the accumulated result
    that reappears at the start of the next chunk (down to 11 chars) and splice
    there; if none is found, join with a newline.
    """
    if not texts:
        return ""

    result = texts[0]
    for text in texts[1:]:
        best_overlap = 0
        upper = min(overlap_chars, len(result), len(text))
        for length in range(upper, 10, -1):
            if result.endswith(text[:length]) or text[:length] in result[-overlap_chars * 2:]:
                best_overlap = length
                break
        if best_overlap > 0:
            idx = result.rfind(text[:best_overlap])
            result = result[:idx] + text if idx >= 0 else result + "\n" + text
        else:
            result = result + "\n" + text
    return result
