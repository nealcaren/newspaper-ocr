from __future__ import annotations

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

    Requires: ``pip install "newspaper-ocr[mineru]"``.
    """

    #: MinerU backend (see :data:`newspaper_ocr._mineru.BACKENDS`).
    backend = "transformers"

    def __init__(self, model: str = _mineru.DEFAULT_MODEL,
                 device: str | None = None, backend: str | None = None,
                 server_url: str | None = None, **kwargs):
        # model_dir / skip_lines are passed by Pipeline for every detector;
        # MinerU is region-only and uses the Hugging Face cache.
        self.backend = backend or self.backend
        self.device = _mineru.resolve_device(self.backend, device)
        self.client = _mineru.get_client(
            model, self.device, backend=self.backend, server_url=server_url
        )

    def detect(self, image: Image.Image) -> PageLayout:
        w, h = image.size
        try:
            blocks = self.client.layout_detect(image)
        except Exception as exc:
            if _device.is_oom(exc):
                _device.free_cache(self.device)
                raise MemoryError(_device.OOM_HINT) from exc
            raise
        regions: list[Region] = []
        for block in blocks:
            nx0, ny0, nx1, ny1 = block.bbox
            x0, y0 = max(0, round(nx0 * w)), max(0, round(ny0 * h))
            x1, y1 = min(w, round(nx1 * w)), min(h, round(ny1 * h))
            if x1 <= x0 or y1 <= y0:
                continue
            regions.append(Region(
                bbox=BBox(x0, y0, x1, y1),
                image=image.crop((x0, y0, x1, y1)),
                label=block.type,
                confidence=1.0,
            ))
        return PageLayout(image=image, regions=regions, width=w, height=h,
                          lines_detected=False, ordered=True)


class MineruVllmDetector(MineruDetector):
    """:class:`MineruDetector` on an in-process vLLM engine (CUDA)."""

    backend = "vllm"


class MineruHttpDetector(MineruDetector):
    """:class:`MineruDetector` against a vLLM server (``MINERU_SERVER_URL``)."""

    backend = "http"
