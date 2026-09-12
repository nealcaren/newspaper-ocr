"""Pathological-repetition detection for VLM recognizer output.

Vision-language models occasionally fall into a loop and emit the same phrase
over and over instead of stopping.  The detection here mirrors the production
GLM-OCR pipeline (reference tag ``2025-03-07-col-fix``): windows are slid across
the *whole* text at multiple offsets and counted with ``str.count``, rather than
only checking patterns anchored at the start of the text.
"""
from __future__ import annotations

# Production defaults (ocr_pipeline.py, tag 2025-03-07-col-fix).
MIN_LEN = 20
MIN_REPS = 5


def _most_repeated(text: str, min_len: int) -> tuple[str | None, int]:
    """Return the most frequently occurring ``min_len``-char window and its count."""
    best: str | None = None
    best_count = 0
    for start in range(0, len(text) - min_len + 1, min_len):
        window = text[start : start + min_len]
        count = text.count(window)
        if count > best_count:
            best, best_count = window, count
    return best, best_count


def has_repetition(
    text: str, min_len: int = MIN_LEN, min_reps: int = MIN_REPS
) -> bool:
    """True if any ``min_len``-char window occurs at least ``min_reps`` times."""
    if len(text) < min_len * min_reps:
        return False
    _, count = _most_repeated(text, min_len)
    return count >= min_reps


def truncate_repetition(text: str, min_len: int = MIN_LEN) -> str:
    """Cut ``text`` just after the second occurrence of its most repeated phrase.

    Keeping two occurrences (rather than one) preserves legitimately repeated
    text — mastheads, running heads, tabular labels — that happens to trip the
    detector.  Text with no repeated window is returned unchanged apart from
    surrounding whitespace.
    """
    phrase, count = _most_repeated(text, min_len)
    if phrase is None or count < 2:
        return text.strip()

    first = text.find(phrase)
    second = text.find(phrase, first + 1)
    if second == -1:
        return text.strip()
    return text[: second + len(phrase)].strip()
