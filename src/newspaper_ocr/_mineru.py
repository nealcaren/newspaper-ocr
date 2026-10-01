"""Shared MinerU2.5 client for the ``mineru`` detector and recognizer.

MinerU2.5 is one 1.2B Qwen2-VL model that does both layout detection and
block recognition, so :class:`~newspaper_ocr.detectors.mineru.MineruDetector`
and :class:`~newspaper_ocr.recognizers.mineru.MineruRecognizer` share a single
loaded copy per (model, device) instead of holding it in memory twice.

Requires: ``pip install "newspaper-ocr[mineru]"``.
"""
from __future__ import annotations

DEFAULT_MODEL = "opendatalab/MinerU2.5-2509-1.2B"

_CLIENTS: dict[tuple[str, str], object] = {}


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_client(model: str = DEFAULT_MODEL, device: str | None = None, **client_kwargs):
    """Return a cached ``MinerUClient`` on the transformers backend.

    CUDA and MPS run bf16 (float32 needs >8 GB on MPS just for layout
    detection); CPU runs float32.  Attention is
    SDPA everywhere: eager attention materializes the vision encoder's full
    attention matrix over the 1036x1036 layout image (~1.8 GB per layer in
    float32), which runs a Mac out of memory.
    """
    try:
        import torch
        from mineru_vl_utils import MinerUClient
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError:
        raise ImportError(
            "MinerU2.5 is not installed. Install with:\n"
            '  pip install "newspaper-ocr[mineru]"'
        )

    from newspaper_ocr._device import prepare_device

    # mineru_vl_utils logs every page's raw layout output at DEBUG through
    # loguru, which prints by default; silence it unless MinerU's own debug
    # switch is on.
    import os
    from loguru import logger
    if not os.environ.get("MINERU_VL_DEBUG_ENABLE"):
        logger.disable("mineru_vl_utils")

    device = prepare_device(device or default_device())
    key = (model, device)
    if key not in _CLIENTS:
        on_cuda = device.startswith("cuda")
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model, use_fast=True)
        client_kwargs.setdefault("use_tqdm", False)
        # The client's default batch_size=0 means unbounded: a dense newspaper
        # page (100+ blocks) then goes through the model in one batch, which
        # exhausts memory on a Mac.  Keep batches small off CUDA.
        client_kwargs.setdefault("batch_size", 16 if on_cuda else 2)
        _CLIENTS[key] = MinerUClient(
            backend="transformers", model=hf_model, processor=processor,
            **client_kwargs,
        )
    return _CLIENTS[key]
