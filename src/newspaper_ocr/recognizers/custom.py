"""Adapter for plugging a plain function in as an OCR device.

The lowest-friction "bring your own OCR" path: write a function that takes a
:class:`PIL.Image.Image` and returns a string, and hand it straight to the
pipeline. No subclassing, no registry edits.

>>> from newspaper_ocr import Pipeline
>>>
>>> def my_ocr(image):
...     # call OpenAI / OpenRouter / a local model / anything
...     return call_my_service(image)
...
>>> Pipeline(recognizer=my_ocr, output="text").run("page.jp2")

:class:`~newspaper_ocr.pipeline.Pipeline` wraps any callable passed as
``recognizer`` or ``fallback`` in :class:`CallableRegionRecognizer`, so you
rarely construct this class directly. Do so only when you want to tune retries
or the loop-detector thresholds.
"""
from __future__ import annotations

from typing import Callable

from PIL import Image

from newspaper_ocr import repetition
from newspaper_ocr.errors import is_timeout
from newspaper_ocr.models import Region
from newspaper_ocr.recognizers.base import RegionRecognizer

#: Placeholder text written when a region exhausts its retries on a timeout.
TIMEOUT_TEXT = "[OCR timeout]"


class CallableRegionRecognizer(RegionRecognizer):
    """Wrap a ``fn(image) -> text`` callable as a region recognizer.

    The wrapped function is called once per detected region with that region's
    cropped image. Exceptions are caught and turned into ``status="timeout"``
    (for anything :func:`~newspaper_ocr.errors.is_timeout` recognises) or
    ``status="error"``, matching the built-in VLM backends, so one flaky call
    can't abort a whole page. Responses that fall into a repeated-phrase loop
    are retried and, if still looping, truncated at the loop boundary.

    Parameters
    ----------
    fn:
        Callable taking a :class:`PIL.Image.Image` and returning the recognized
        text as a string.
    label:
        Optional name recorded on ``Region.engine`` for auditing; defaults to
        the callable's ``__name__``.
    max_retries:
        Extra attempts after the first on error, timeout, or a looping response.
    repetition_min_len / repetition_min_reps:
        Loop-detector thresholds, shared with the other VLM backends.
    """

    def __init__(
        self,
        fn: Callable[[Image.Image], str],
        label: str | None = None,
        max_retries: int = 1,
        repetition_min_len: int = repetition.MIN_LEN,
        repetition_min_reps: int = repetition.MIN_REPS,
    ):
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        self.fn = fn
        self.label = label or getattr(fn, "__name__", "custom")
        self.max_retries = max_retries
        self.repetition_min_len = repetition_min_len
        self.repetition_min_reps = repetition_min_reps

    def recognize(self, region: Region) -> Region:
        for attempt in range(self.max_retries + 1):
            try:
                text = (self.fn(region.image) or "").strip()
            except Exception as exc:
                if attempt < self.max_retries:
                    continue
                timed_out = is_timeout(exc)
                region.text = TIMEOUT_TEXT if timed_out else ""
                region.status = "timeout" if timed_out else "error"
                return region

            if not repetition.has_repetition(
                text, self.repetition_min_len, self.repetition_min_reps
            ):
                region.text = text
                region.status = "ok"
                return region
            if attempt < self.max_retries:
                continue
            region.text = repetition.truncate_repetition(
                text, self.repetition_min_len
            )
            region.status = "repetition"
            return region
        return region
