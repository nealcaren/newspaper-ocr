"""Device helpers for the local model backends — chiefly Apple Silicon safety.

On a Mac the GPU (MPS) shares system memory, and PyTorch's default MPS limit
lets a process grow to ~1.7x the GPU's recommended working set — far enough to
push the whole machine into swap.  :func:`prepare_device` caps MPS allocations
so an oversized batch fails with a catchable out-of-memory error instead.

Override the cap with ``NEWSPAPER_OCR_MPS_MEMORY_FRACTION`` (a fraction of
macOS's recommended GPU working set; ``0`` removes the limit).  If
``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` is set, PyTorch's own limit is left alone.
"""
from __future__ import annotations

import os

#: Default MPS cap, as a fraction of the recommended GPU working set (about
#: 13-14 GB on a 36 GB Mac).
DEFAULT_MPS_FRACTION = 0.5

_mps_limited = False


def prepare_device(device: str) -> str:
    """Apply per-device safety settings before a model is moved to *device*."""
    if device.startswith("mps"):
        limit_mps_memory()
    return device


def limit_mps_memory(fraction: float | None = None) -> None:
    """Cap this process's MPS allocations (once per process)."""
    global _mps_limited
    if _mps_limited or "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
        return
    import torch

    if fraction is None:
        fraction = float(os.environ.get("NEWSPAPER_OCR_MPS_MEMORY_FRACTION",
                                        DEFAULT_MPS_FRACTION))
    torch.mps.set_per_process_memory_fraction(fraction)
    _mps_limited = True


def is_oom(exc: BaseException) -> bool:
    """True if *exc* is a GPU out-of-memory error (CUDA or MPS)."""
    name = type(exc).__name__
    return name == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def free_cache(device: str) -> None:
    """Release cached GPU memory back to the system (no-op on CPU)."""
    try:
        import torch
    except ImportError:
        return
    if device.startswith("mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


OOM_HINT = (
    "Ran out of GPU memory. On a Mac, newspaper-ocr caps MPS memory at "
    f"{DEFAULT_MPS_FRACTION:.0%} of the GPU working set to keep the machine "
    "responsive; raise it with NEWSPAPER_OCR_MPS_MEMORY_FRACTION (e.g. 0.8), "
    "close other apps, run with device='cpu', or use a CUDA machine."
)
