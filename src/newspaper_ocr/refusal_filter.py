"""Drop the refusals and image descriptions a VLM writes instead of a reading.

Given a tiny, blank or blurry crop, chat-tuned VLMs answer about the image
rather than transcribing it: "There is no legible text in this image", "I'm
sorry, but I can't read the text in this image", "The image has been rotated
90 degrees clockwise".  On NewsBench (15 pages) gemini-3.5-flash-lite wrote
162 words of this, deepseek-v4.1-flash 355 and gpt-5.6-luna 26, all scored as
page text.

:func:`drop_refusals` removes each sentence that talks about the image or the
transcription task.  A sentence counts when it names the image ("this image",
"the picture", but not "the image of God") and also uses a word about seeing or
reading it (text, legible, blurry, rotated, transcribe ...), or when it is one
of a few fixed refusal phrases ("no legible text", "unable to transcribe").  A
region left with no real text is blanked and its status set to
``hallucination``; otherwise only the matching sentences are cut.  The original
read is kept in ``Region.text_primary``.
"""
from __future__ import annotations

import re

from newspaper_ocr.models import PageLayout

# "the image", "this picture", "provided crop" -- but not "the image of God".
_IMAGE = re.compile(
    r"\b(?:the|this|that|provided|given|an?|your)\s+"
    r"(?:image|picture|photo|photograph|crop|scan)\b(?!\s+of\b)",
    re.I,
)
_CUE = re.compile(
    r"\b(?:text|legible|readable|visible|blank|black|blurr\w*|blurred|rotated|"
    r"flipped|degrees|transcri\w*|resolution|recogni[sz]e|contains?|shows?|"
    r"appears?|depicts?|provide|clearer|larger|characters|words)\b",
    re.I,
)
_PHRASES = re.compile(
    r"\bno\s+(?:discernible|legible|readable|visible|transcribable|recogni[sz]able)"
    r"\s+text\b"
    r"|\bno\s+text\s+(?:is\s+)?(?:visible|present|found|content|to\s+transcribe)\b"
    r"|\b(?:unable|not\s+able)\s+to\s+(?:transcribe|make\s+out|recogni[sz]e)\b"
    r"|\b(?:cannot|can['’]?t|can\s+not|could\s+not|couldn['’]?t)\s+"
    r"(?:transcribe|make\s+out\s+any|read\s+any\s+text|read\s+the\s+text)\b"
    r"|\bI['’]?m\s+sorry,?\s+but\s+I\b",
    re.I,
)
# Sentence ends and line breaks.  A description tacked on with a dash ("no
# legible text in this image -- it appears to be ...") goes with its sentence.
_SPLIT = re.compile(r"(?<=[.!?:])\s+|\n+")

#: Blank the region when fewer than this many letters/digits survive.
MIN_KEPT = 4


def _is_refusal(sentence: str) -> bool:
    if _PHRASES.search(sentence):
        return True
    return bool(_IMAGE.search(sentence) and _CUE.search(sentence))


def strip_refusals(text: str) -> tuple[str, bool]:
    """Return ``text`` with refusal sentences cut, and whether any were."""
    if not text:
        return text, False
    lines, cut = [], False
    for line in text.split("\n"):
        parts = _SPLIT.split(line)
        kept = [p for p in parts if not _is_refusal(p)]
        if len(kept) != len(parts):
            cut = True
            line = " ".join(kept).strip()
            if not line:
                continue
        lines.append(line)
    return ("\n".join(lines).strip() if cut else text), cut


def drop_refusals(layout: PageLayout, min_kept: int = MIN_KEPT) -> PageLayout:
    """Cut refusal sentences; blank regions that held nothing else."""
    for r in layout.regions:
        text, cut = strip_refusals(r.text or "")
        if not cut:
            continue
        r.text_primary = r.text
        if sum(c.isalnum() for c in text) < min_kept:
            r.text, r.status = "", "hallucination"
        else:
            r.text = text
    return layout
