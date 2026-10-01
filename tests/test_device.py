"""Tests for the Apple Silicon memory guard in newspaper_ocr._device."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from newspaper_ocr import _device  # noqa: E402


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", seen.append)
    monkeypatch.setattr(_device, "_mps_limited", False)
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("NEWSPAPER_OCR_MPS_MEMORY_FRACTION", raising=False)
    return seen


def test_mps_capped_by_default_once(calls):
    assert _device.prepare_device("mps") == "mps"
    _device.prepare_device("mps")
    assert calls == [_device.DEFAULT_MPS_FRACTION]


def test_cuda_and_cpu_untouched(calls):
    _device.prepare_device("cuda")
    _device.prepare_device("cpu")
    assert calls == []


def test_env_override(calls, monkeypatch):
    monkeypatch.setenv("NEWSPAPER_OCR_MPS_MEMORY_FRACTION", "0.8")
    _device.prepare_device("mps")
    assert calls == [0.8]


def test_respects_pytorch_watermark(calls, monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
    _device.prepare_device("mps")
    assert calls == []


def test_is_oom():
    assert _device.is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 8 GB)"))
    assert _device.is_oom(torch.OutOfMemoryError("CUDA out of memory"))
    assert not _device.is_oom(ValueError("bad"))
