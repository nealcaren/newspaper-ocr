from __future__ import annotations

from PIL import Image

from newspaper_ocr import _device, _mineru, repetition
from newspaper_ocr.models import Region
from newspaper_ocr.recognizers.base import RegionRecognizer

# MinerU block types it never sends to the model on its own pages: containers
# (whose children are read instead) and pictures.
_SKIP_TYPES = {"image", "chart", "list", "equation_block", "image_block"}

_FAILED = object()


class MineruRecognizer(RegionRecognizer):
    """MinerU2.5 block recognition.

    A region whose label is a MinerU block type (from the ``mineru`` detector)
    is read with that type's prompt — tables as tables, equations as formulas —
    and MinerU's own picture and container blocks are skipped, as in MinerU's
    pipeline.  Any other label (e.g. a DocLayout-YOLO hole, including its
    ``figure`` boxes, which on newspapers are often ads full of text) is read
    as text.

    :meth:`recognize_regions` reads a whole page's regions in one batched call;
    :class:`~newspaper_ocr.pipeline.Pipeline` uses it when present.

    Requires: ``pip install "newspaper-ocr[mineru]"``.
    """

    mode = "region"
    #: MinerU backend (see :data:`newspaper_ocr._mineru.BACKENDS`).
    backend = "transformers"

    def __init__(
        self,
        model: str = _mineru.DEFAULT_MODEL,
        device: str | None = None,
        repetition_min_len: int = repetition.MIN_LEN,
        repetition_min_reps: int = repetition.MIN_REPS,
        backend: str | None = None,
        server_url: str | None = None,
        **kwargs,
    ):
        self.backend = backend or self.backend
        self.device = _mineru.resolve_device(self.backend, device)
        self.client = _mineru.get_client(
            model, self.device, backend=self.backend, server_url=server_url
        )
        self.repetition_min_len = repetition_min_len
        self.repetition_min_reps = repetition_min_reps
        from mineru_vl_utils.structs import BLOCK_TYPES
        self._block_types = BLOCK_TYPES

    def _block_type(self, label: str) -> str:
        return label if label in self._block_types else "text"

    def recognize(self, region: Region) -> Region:
        return self.recognize_regions(None, [region])[0]

    def recognize_regions(self, page_image: Image.Image | None,
                          regions: list[Region]) -> list[Region]:
        """Read all *regions* of a page in one batch; returns them updated."""
        todo = [r for r in regions
                if r.image is not None and self._block_type(r.label) not in _SKIP_TYPES]
        todo_ids = {id(r) for r in todo}
        for r in regions:
            if id(r) not in todo_ids:
                r.text, r.status = "", "ok"
        if not todo:
            return regions

        try:
            results = self._extract(todo)
        except Exception:
            # One bad crop (often an out-of-memory on a huge region) shouldn't
            # sink the page: retry one region at a time and mark only the
            # failures, which the pipeline's fallback ladder can then pick up.
            _device.free_cache(self.device)
            results = []
            for r in todo:
                try:
                    results.extend(self._extract([r]))
                except Exception:
                    _device.free_cache(self.device)
                    results.append(_FAILED)
        finally:
            _device.free_cache(self.device)

        for r, res in zip(todo, results):
            if res is _FAILED:
                r.text, r.status = "", "error"
                continue
            text = (str(res) if res is not None else "").strip()
            if repetition.has_repetition(
                text, self.repetition_min_len, self.repetition_min_reps
            ):
                r.text = repetition.truncate_repetition(text, self.repetition_min_len)
                r.status = "repetition"
            else:
                r.text, r.status = text, "ok"
        return regions

    def _extract(self, regions: list[Region]) -> list:
        return self.client.batch_content_extract(
            [r.image.convert("RGB") for r in regions],
            [self._block_type(r.label) for r in regions],
        )


class MineruVllmRecognizer(MineruRecognizer):
    """:class:`MineruRecognizer` on an in-process vLLM engine (CUDA)."""

    backend = "vllm"


class MineruHttpRecognizer(MineruRecognizer):
    """:class:`MineruRecognizer` against a vLLM server (``MINERU_SERVER_URL``)."""

    backend = "http"
