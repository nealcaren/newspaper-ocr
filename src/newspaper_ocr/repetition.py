"""Pathological-repetition detection for VLM recognizer output.

Vision-language models occasionally fall into a loop and emit the same phrase
over and over instead of stopping.  The detection here mirrors the production
GLM-OCR pipeline (reference tag ``2025-03-07-col-fix``): windows are slid across
the *whole* text at every offset and counted with ``str.count``, rather than only
checking patterns anchored at the start of the text.

Both passes are O(n^2) in the length of a region's text, but ``str.count`` runs
in C and a region tops out at a few thousand characters, so the cost is
immaterial next to the VLM call that produced the text.
"""
from __future__ import annotations

# Production defaults (ocr_pipeline.py, tag 2025-03-07-col-fix).
MIN_LEN = 20
MIN_REPS = 5


def _most_repeated(text: str, min_len: int) -> tuple[str | None, int]:
    """Return the most frequently occurring ``min_len``-char window and its count.

    Windows are tried at *every* offset, not just multiples of ``min_len``.  The
    stride matters for truncation: in a looping region every window inside the
    loop occurs the same number of times, so the winner is whichever offset is
    reached first, and a coarse stride picks one that starts mid-phrase — which
    moves the cut point off the phrase boundary.  Ties therefore resolve to the
    earliest offset, matching production.
    """
    best: str | None = None
    best_count = 0
    for start in range(len(text) - min_len + 1):
        window = text[start : start + min_len]
        count = text.count(window)
        if count > best_count:
            best, best_count = window, count
    return best, best_count


def has_repetition(
    text: str, min_len: int = MIN_LEN, min_reps: int = MIN_REPS
) -> bool:
    """True if any ``min_len``-char window occurs at least ``min_reps`` times.

    Note that ``str.count`` counts *non-overlapping* occurrences, so a phrase
    shorter than ``min_len`` is undercounted.  That's production's behaviour too,
    and it's the conservative direction: a real loop of a short phrase still
    repeats far more than ``min_reps`` times before it trips the detector.
    """
    if len(text) < min_len * min_reps:
        return False
    # Short-circuit on the first qualifying window rather than scanning for the
    # maximum — detection only needs to know that one exists.
    for start in range(len(text) - min_len + 1):
        if text.count(text[start : start + min_len]) >= min_reps:
            return True
    return False


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
