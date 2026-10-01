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


def resolve_device(backend: str, device: str | None = None) -> str:
    """Device for *backend*: transformers picks one; vLLM is CUDA; http is remote."""
    if backend == "transformers":
        return device or default_device()
    if backend not in BACKENDS:
        raise ValueError(f"MinerU backend must be one of {BACKENDS}; got {backend!r}")
    return {"vllm": "cuda", "http": "remote"}[backend]


#: Backends: ``transformers`` (default; CUDA, MPS or CPU), ``vllm`` (in-process
#: vLLM engine; CUDA only, several times faster on a full page of blocks) and
#: ``http`` (a running ``vllm serve`` / ``mineru-vllm-server``, at *server_url*).
BACKENDS = ("transformers", "vllm", "http")

#: Fraction of GPU memory the in-process vLLM engine may claim.  vLLM grabs it
#: all up front for its KV cache, so leave room for a hole-fill detector
#: (DocLayout-YOLO needs ~2 GB) sharing the card.
VLLM_GPU_MEMORY = 0.75


def get_client(
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    backend: str = "transformers",
    server_url: str | None = None,
    **client_kwargs,
):
    """Return a cached ``MinerUClient`` for *backend*.

    The detector and recognizer share one client per (model, device, backend,
    server) so a page never holds two copies of the model.

    ``http`` needs *server_url* or ``MINERU_SERVER_URL``; ``vllm`` reads
    ``NEWSPAPER_OCR_VLLM_GPU_MEMORY`` to override :data:`VLLM_GPU_MEMORY`.
    """
    import os

    if backend not in BACKENDS:
        raise ValueError(f"MinerU backend must be one of {BACKENDS}; got {backend!r}")
    try:
        from mineru_vl_utils import MinerUClient
    except ImportError:
        raise ImportError(
            "MinerU2.5 is not installed. Install with:\n"
            '  pip install "newspaper-ocr[mineru]"'
        )

    # mineru_vl_utils logs every page's raw layout output at DEBUG through
    # loguru, which prints by default; silence it unless MinerU's own debug
    # switch is on.
    from loguru import logger
    if not os.environ.get("MINERU_VL_DEBUG_ENABLE"):
        logger.disable("mineru_vl_utils")
    client_kwargs.setdefault("use_tqdm", False)

    if backend == "http":
        server_url = server_url or os.environ.get("MINERU_SERVER_URL")
        if not server_url:
            raise ValueError(
                "The http MinerU backend needs a server: pass server_url= or "
                "set MINERU_SERVER_URL (e.g. http://localhost:30000)."
            )
        key = (model, "http", backend, server_url)
        if key not in _CLIENTS:
            _CLIENTS[key] = MinerUClient(
                backend="http-client", server_url=server_url, **client_kwargs
            )
        return _CLIENTS[key]

    if backend == "vllm":
        key = (model, "cuda", backend, None)
        if key not in _CLIENTS:
            _CLIENTS[key] = MinerUClient(
                backend="vllm-engine", vllm_llm=_vllm_engine(model), **client_kwargs
            )
        return _CLIENTS[key]

    return _transformers_client(model, device, client_kwargs)


def _vllm_engine(model: str):
    import os

    try:
        from mineru_vl_utils import MinerULogitsProcessor
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "vLLM is not installed. Install with:\n"
            '  pip install "newspaper-ocr[mineru-vllm]"'
        )
    memory = float(os.environ.get("NEWSPAPER_OCR_VLLM_GPU_MEMORY", VLLM_GPU_MEMORY))
    # vLLM's default FlashInfer sampler JIT-compiles a CUDA kernel on first
    # use, which fails on cluster nodes without a CUDA toolkit ("Could not
    # find nvcc"). PyTorch-native sampling needs no compiler and costs little
    # at MinerU's short outputs; set the variable to 1 to opt back in.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # MinerU's no-repeat-ngram processor is what its own vLLM pipeline uses
    # to stop the model looping on dense text.
    #
    # Prefix caching is off: with it on, layout detection on a page whose
    # prompt was already cached comes out different from the cold run, and
    # sometimes broken (a NewsBench page went from 203 blocks to 681 on every
    # cached repeat, its score from 0.91 to 0.52). Off, repeats are identical
    # to the cold run. Batch pages are distinct images, so caching saved
    # little anyway.
    return LLM(
        model=model,
        gpu_memory_utilization=memory,
        logits_processors=[MinerULogitsProcessor],
        enable_prefix_caching=False,
    )


def _transformers_client(model: str, device: str | None, client_kwargs: dict):
    """Transformers backend.

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

    device = prepare_device(device or default_device())
    key = (model, device, "transformers", None)
    if key not in _CLIENTS:
        on_cuda = device.startswith("cuda")
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model, use_fast=True)
        # The client's default batch_size=0 means unbounded: a dense newspaper
        # page (100+ blocks) then goes through the model in one batch, which
        # exhausts memory on a Mac.  Keep batches small off CUDA.
        client_kwargs.setdefault("batch_size", 16 if on_cuda else 2)
        _CLIENTS[key] = MinerUClient(
            backend="transformers", model=hf_model, processor=processor,
            **client_kwargs,
        )
    return _CLIENTS[key]
