"""Exception types shared across recognizers."""
from __future__ import annotations


class OcrTimeout(Exception):
    """A single region exceeded the recognizer's wall-clock budget."""


def is_timeout(exc: BaseException) -> bool:
    """True if ``exc`` represents a wall-clock timeout.

    Recognises our own :class:`OcrTimeout`, the builtin ``TimeoutError``, and
    third-party timeouts (e.g. ``httpx.TimeoutException``) by class name, so we
    don't have to import optional backends just to classify a failure.
    """
    if isinstance(exc, (OcrTimeout, TimeoutError)):
        return True
    return any("Timeout" in cls.__name__ for cls in type(exc).__mro__)
