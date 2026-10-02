"""Drop CJK text a VLM invented on a Latin-script page.

Given a tiny or blank crop, MinerU2.5 (trained heavily on Chinese documents)
often answers in Chinese: a single ``信`` or ``表``, an exam question, a stock
code.  On 200 pages of 1900-1932 African American newspapers, 130 pages came
back with CJK characters (5,776 in all) from 685 regions, and the papers have
no CJK text at all (issue #30).

:func:`drop_cjk_hallucinations` decides from the page's own text whether the
page is Latin-script: whether Latin letters outnumber CJK characters across all
of its regions.  If so, a region whose letters are mostly CJK is blanked and its
status set to ``hallucination``, and stray CJK characters in the remaining
regions are removed.  The original read is kept in ``Region.text_primary``.  A
page that is really CJK is left alone.
"""
from __future__ import annotations

import re

from newspaper_ocr.models import PageLayout

# Ideographs, kana, hangul and CJK punctuation.  Fullwidth forms (U+FF00-FFEF)
# are left out: they include fullwidth Latin letters and digits.
_CJK = re.compile(
    "[　-〿぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]"
)
_LATIN = re.compile("[A-Za-zÀ-ɏ]")

#: A region at or above this CJK share of its letters is blanked.
REGION_CJK_MAX = 0.3


def _counts(text: str) -> tuple[int, int]:
    return len(_CJK.findall(text)), len(_LATIN.findall(text))


def drop_cjk_hallucinations(layout: PageLayout,
                            region_cjk_max: float = REGION_CJK_MAX) -> PageLayout:
    """Blank mostly-CJK regions and strip stray CJK on a Latin-script page."""
    cjk = latin = 0
    for r in layout.regions:
        c, l = _counts(r.text or "")
        cjk += c
        latin += l
    if cjk == 0 or cjk >= latin:
        return layout

    for r in layout.regions:
        c, l = _counts(r.text or "")
        if c == 0:
            continue
        r.text_primary = r.text
        if c >= region_cjk_max * (c + l):
            r.text, r.status = "", "hallucination"
        else:
            r.text = re.sub(r"[ \t]{2,}", " ", _CJK.sub("", r.text)).strip()
    return layout
