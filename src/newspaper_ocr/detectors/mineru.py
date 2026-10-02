from __future__ import annotations

import numpy as np
from PIL import Image

from newspaper_ocr import _device, _mineru
from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.models import BBox, PageLayout, Region


class MineruDetector(Detector):
    """MinerU2.5 layout detection (``MinerUClient.layout_detect``).

    MinerU returns blocks already in reading order, and its order is excellent
    on newspaper pages, so the layout is marked ``ordered`` and layout
    post-processing keeps it rather than re-sorting.  It does miss isolated
    blocks (a poem stanza, an ad, a side column); pair it with a hole-fill
    detector to recover those::

        Pipeline(detector="mineru", hole_fill_detector="doclayout_yolo",
                 recognizer="mineru")

    Labels are MinerU block types (``text``, ``title``, ``image``, ``header``,
    ...).  MinerU gives no per-block score, so every block gets confidence 1.0.

    Two failures get a second try on smaller tiles (``split_retry``, on by
    default; issue #30):

    * **Truncation.**  The layout is generated as text, and the model's 16,384
      tokens fit about 680 blocks.  Dense small type can be boxed line by line,
      and then the page stops partway through: on a 1905 *New York Age* page the
      last two columns got no boxes at all.  A tile with ``SPLIT_MIN_BLOCKS``
      or more blocks is retried.
    * **One big picture.**  A tile boxed mostly as one ``image`` (more than
      ``SPLIT_PICTURE_FRAC`` of it).  This misfire can strike a tile of any
      size: on a 1914 *New York Age* page it hit the page, then its left half,
      then that half's top quarter.

    A suspect tile is cut in two, alternating vertical and horizontal cuts
    (left/right first), down to ``SPLIT_MAX_DEPTH`` levels.  The pieces overlap
    by ``SPLIT_OVERLAP`` of the tile, and a box belongs to the piece holding its
    center.  Left/right pieces are read left first; a bottom piece's blocks go
    in after the top piece's blocks in their own column (see
    :func:`~newspaper_ocr.layout_processor.insert_in_order`).  The pieces
    replace the tile's own layout only if their text boxes cover more of it.

    Requires: ``pip install "newspaper-ocr[mineru]"``.
    """

    #: MinerU backend (see :data:`newspaper_ocr._mineru.BACKENDS`).
    backend = "transformers"

    SPLIT_MIN_BLOCKS = 500
    SPLIT_PICTURE_FRAC = 0.5
    SPLIT_OVERLAP = 0.05
    SPLIT_MAX_DEPTH = 3
    _PICTURES = frozenset({"image", "image_block"})

    def __init__(self, model: str = _mineru.DEFAULT_MODEL,
                 device: str | None = None, backend: str | None = None,
                 server_url: str | None = None, split_retry: bool = True,
                 **kwargs):
        # model_dir / skip_lines are passed by Pipeline for every detector;
        # MinerU is region-only and uses the Hugging Face cache.
        self.backend = backend or self.backend
        self.device = _mineru.resolve_device(self.backend, device)
        self.client = _mineru.get_client(
            model, self.device, backend=self.backend, server_url=server_url
        )
        self.split_retry = split_retry
        #: Tiles the most recent page's layout came from (1 = not split).
        self.last_tiles = 1

    def detect(self, image: Image.Image) -> PageLayout:
        w, h = image.size
        max_depth = self.SPLIT_MAX_DEPTH if self.split_retry else 0
        regions, self.last_tiles = self._tiled(image, (0, 0, w, h), 0, max_depth)
        return PageLayout(image=image, regions=regions, width=w, height=h,
                          lines_detected=False, ordered=True)

    def _tiled(self, image: Image.Image, tile: tuple[int, int, int, int],
               depth: int, max_depth: int) -> tuple[list[Region], int]:
        """Layout of *tile*, split further while it looks broken; (regions, tiles)."""
        regions = self._layout(image, tile)
        if depth >= max_depth or not self._suspect(regions, tile):
            return regions, 1
        x0, y0, x1, y1 = tile
        axis = depth % 2  # 0: left | right, 1: top / bottom
        lo, hi = (x0, x1) if axis == 0 else (y0, y1)
        cut = (lo + hi) // 2
        m = round(self.SPLIT_OVERLAP * (hi - lo))
        first = (x0, y0, cut + m, y1) if axis == 0 else (x0, y0, x1, cut + m)
        second = (cut - m, y0, x1, y1) if axis == 0 else (x0, cut - m, x1, y1)
        a, na = self._tiled(image, first, depth + 1, max_depth)
        b, nb = self._tiled(image, second, depth + 1, max_depth)
        a, b = self._join(image, a, b, axis, cut, m)
        if axis == 0:
            pieces = a + b
        else:
            from newspaper_ocr.layout_processor import insert_in_order
            pieces = insert_in_order(a, b)
        if self._text_cover(pieces, tile) > self._text_cover(regions, tile):
            return pieces, na + nb
        return regions, 1

    @staticmethod
    def _join(image: Image.Image, a: list[Region], b: list[Region], axis: int,
              cut: int, m: int) -> tuple[list[Region], list[Region]]:
        """Settle the boxes of two pieces cut at *cut* (overlapping by *m*).

        A block straddling the cut is boxed in both pieces, each box clipped at
        its piece's edge: such a pair becomes one box, kept with the first
        piece.  Every other box goes to the piece holding its center, which
        drops the second copy of a block inside the overlap.
        """
        def span(r, ax):
            b = r.bbox
            return (b.x0, b.x1) if ax == 0 else (b.y0, b.y1)

        tol = max(2, m // 2)
        joined: set[int] = set()
        keep_a: list[Region] = []
        for r in a:
            r_lo, r_hi = span(r, axis)
            if r_hi >= cut + m - tol:
                o_lo, o_hi = span(r, 1 - axis)
                for s in b:
                    s_lo, s_hi = span(s, axis)
                    p_lo, p_hi = span(s, 1 - axis)
                    shared = min(o_hi, p_hi) - max(o_lo, p_lo)
                    if (id(s) not in joined and s_lo <= cut - m + tol
                            and shared > 0.5 * min(o_hi - o_lo, p_hi - p_lo)):
                        rb, sb = r.bbox, s.bbox
                        r.bbox = BBox(min(rb.x0, sb.x0), min(rb.y0, sb.y0),
                                      max(rb.x1, sb.x1), max(rb.y1, sb.y1))
                        r.image = image.crop(r.bbox.to_tuple())
                        joined.add(id(s))
                        keep_a.append(r)
                        break
                else:
                    if r_lo + r_hi < 2 * cut:
                        keep_a.append(r)
            elif r_lo + r_hi < 2 * cut:
                keep_a.append(r)
        keep_b = [s for s in b if id(s) not in joined and sum(span(s, axis)) >= 2 * cut]
        return keep_a, keep_b

    def _layout(self, image: Image.Image, tile: tuple[int, int, int, int]) -> list[Region]:
        """MinerU's blocks for the *tile* of *image*, in page coordinates."""
        tx0, ty0, tx1, ty1 = tile
        crop = image if tile == (0, 0, *image.size) else image.crop(tile)
        tw, th = tx1 - tx0, ty1 - ty0
        try:
            blocks = self.client.layout_detect(crop)
        except Exception as exc:
            if _device.is_oom(exc):
                _device.free_cache(self.device)
                raise MemoryError(_device.OOM_HINT) from exc
            raise
        regions: list[Region] = []
        for block in blocks:
            nx0, ny0, nx1, ny1 = block.bbox
            x0, y0 = tx0 + max(0, round(nx0 * tw)), ty0 + max(0, round(ny0 * th))
            x1, y1 = tx0 + min(tw, round(nx1 * tw)), ty0 + min(th, round(ny1 * th))
            if x1 <= x0 or y1 <= y0:
                continue
            regions.append(Region(
                bbox=BBox(x0, y0, x1, y1),
                image=image.crop((x0, y0, x1, y1)),
                label=block.type,
                confidence=1.0,
            ))
        return regions

    def _suspect(self, regions: list[Region], tile: tuple[int, int, int, int]) -> bool:
        if len(regions) >= self.SPLIT_MIN_BLOCKS:
            return True
        area = (tile[2] - tile[0]) * (tile[3] - tile[1])
        return any(r.label in self._PICTURES
                   and r.bbox.width * r.bbox.height > self.SPLIT_PICTURE_FRAC * area
                   for r in regions)

    def _text_cover(self, regions: list[Region], tile: tuple[int, int, int, int],
                    cell: int = 8) -> float:
        """Fraction of *tile* under non-picture boxes, on a coarse grid."""
        x0, y0, x1, y1 = tile
        mask = np.zeros(((y1 - y0) // cell + 1, (x1 - x0) // cell + 1), dtype=bool)
        for r in regions:
            if r.label not in self._PICTURES:
                b = r.bbox
                mask[max(0, b.y0 - y0) // cell:max(0, b.y1 - y0) // cell + 1,
                     max(0, b.x0 - x0) // cell:max(0, b.x1 - x0) // cell + 1] = True
        return float(mask.mean())


class MineruVllmDetector(MineruDetector):
    """:class:`MineruDetector` on an in-process vLLM engine (CUDA)."""

    backend = "vllm"


class MineruHttpDetector(MineruDetector):
    """:class:`MineruDetector` against a vLLM server (``MINERU_SERVER_URL``)."""

    backend = "http"
