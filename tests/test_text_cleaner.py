"""Tests for TextCleaner post-processing stage."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from newspaper_ocr.models import BBox, Line, PageLayout, Region
from newspaper_ocr.text_cleaner import TextCleaner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line(text: str, y0: int, y1: int, x0: int = 0, x1: int = 400) -> Line:
    """Create a Line with realistic bbox and a tiny dummy image."""
    img = Image.fromarray(np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8))
    return Line(bbox=BBox(x0, y0, x1, y1), image=img, text=text)


def _clean(lines: list[Line]) -> str:
    """Run TextCleaner on a synthetic region and return the text."""
    page_img = Image.fromarray(np.zeros((600, 400, 3), dtype=np.uint8))
    region_img = page_img.copy()
    region = Region(bbox=BBox(0, 0, 400, 600), image=region_img, label="article", lines=lines)
    layout = PageLayout(image=page_img, regions=[region], width=400, height=600)
    cleaner = TextCleaner(enabled=True)
    result = cleaner.clean(layout)
    return result.regions[0].text


# ---------------------------------------------------------------------------
# Dehyphenation
# ---------------------------------------------------------------------------

def test_dehyphenation():
    """'com-' + 'plete the' → 'complete the'"""
    lines = [
        _line("com-", 0, 20),
        _line("plete the", 22, 42),
    ]
    text = _clean(lines)
    assert "complete the" in text
    assert "-" not in text


def test_dehyphenation_joins_without_space():
    """Dehyphenated join must have no space between the word parts."""
    lines = [
        _line("impor-", 0, 20),
        _line("tant news", 22, 42),
    ]
    text = _clean(lines)
    assert "important" in text


def test_dehyphenation_requires_lowercase_next():
    """Hyphen before an uppercase letter is NOT dehyphenated."""
    lines = [
        _line("New-", 0, 20),
        _line("York is great", 22, 42),
    ]
    text = _clean(lines)
    # Should keep the hyphen / treat as continuation, not strip it.
    assert "New" in text
    assert "York" in text


# ---------------------------------------------------------------------------
# Line continuation
# ---------------------------------------------------------------------------

def test_continuation_join():
    """Lines without terminal punctuation are joined with a space."""
    lines = [
        _line("The quick brown", 0, 20),
        _line("fox jumped over", 22, 42),
    ]
    text = _clean(lines)
    assert "The quick brown fox jumped over" in text


def test_multiple_continuations():
    """Three continuation lines produce one paragraph."""
    lines = [
        _line("First line of", 0, 20),
        _line("a very long paragraph", 22, 42),
        _line("that keeps going on.", 44, 64),
    ]
    text = _clean(lines)
    assert "First line of a very long paragraph that keeps going on." in text


# ---------------------------------------------------------------------------
# Terminal punctuation blocks continuation
# ---------------------------------------------------------------------------

def test_terminal_punct_period_no_continuation():
    """A line ending with '.' does not continue into the next line."""
    lines = [
        _line("End of sentence.", 0, 20),
        _line("New sentence here.", 22, 42),
    ]
    text = _clean(lines)
    # When next line starts uppercase after terminal punct → block break.
    assert "End of sentence." in text
    assert "New sentence here." in text


def test_terminal_punct_question_mark():
    """A line ending with '?' blocks continuation when next starts uppercase."""
    lines = [
        _line("Did you see that?", 0, 20),
        _line("Yes I did.", 22, 42),
    ]
    text = _clean(lines)
    assert "Did you see that?" in text
    assert "Yes I did." in text


def test_uppercase_after_period_new_paragraph():
    """Period + uppercase next line = new paragraph (separate blocks)."""
    lines = [
        _line("The story ended.", 0, 20),
        _line("Another story began.", 22, 42),
    ]
    text = _clean(lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    assert len(paragraphs) == 2


# ---------------------------------------------------------------------------
# Semantic dash preservation
# ---------------------------------------------------------------------------

def test_semantic_dash_preserved():
    """Spaced dash (' -') is semantic and preserved."""
    cleaner = TextCleaner()
    # A spaced dash is unambiguously semantic.
    assert cleaner._is_semantic_dash("something -")
    assert cleaner._is_semantic_dash("something --")


def test_plain_hyphen_not_semantic():
    """A plain trailing hyphen (no space, no em-dash) is NOT semantic."""
    cleaner = TextCleaner()
    # Plain word-break hyphens like "an-", "com-", "impor-" are typographic.
    assert not cleaner._is_semantic_dash("an-")
    assert not cleaner._is_semantic_dash("com-")
    assert not cleaner._is_semantic_dash("impor-")


def test_em_dash_preserved():
    """Lines ending with -- (double dash) are treated as semantic."""
    cleaner = TextCleaner()
    assert cleaner._is_semantic_dash("something--")
    assert cleaner._is_semantic_dash("something\u2014")


def test_spaced_dash_preserved():
    """Lines ending with ' -' (space then dash) are semantic."""
    cleaner = TextCleaner()
    assert cleaner._is_semantic_dash("something -")


def test_semantic_dash_not_dehyphenated():
    """A spaced semantic dash is not removed even when next line is lowercase."""
    lines = [
        _line("something -", 0, 20),
        _line("or other text", 22, 42),
    ]
    text = _clean(lines)
    # Spaced dash is semantic; it must not be stripped.
    assert "something -" in text


# ---------------------------------------------------------------------------
# Block breaks
# ---------------------------------------------------------------------------

def test_block_break_on_gap():
    """Large vertical gap (> 1.5x median line height) starts a new paragraph."""
    lines = [
        _line("First paragraph line.", 0, 20),
        # Gap: y0=80, previous y1=20 → gap=60, median height≈20, 60 > 1.5*20 ✓
        _line("Second paragraph line.", 80, 100),
    ]
    text = _clean(lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    assert len(paragraphs) == 2
    assert paragraphs[0] == "First paragraph line."
    assert paragraphs[1] == "Second paragraph line."


def test_block_break_on_column_shift():
    """Significant x-origin shift starts a new paragraph."""
    lines = [
        # Column 1: x0=0
        _line("Left column text.", 0, 20, x0=0, x1=180),
        # Column 2: x0=220 → shift=220, page_width≈220, 220/220=1.0 > 0.15 ✓
        _line("Right column text.", 22, 42, x0=220, x1=400),
    ]
    text = _clean(lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    assert len(paragraphs) == 2


def test_no_block_break_small_gap():
    """Small gap between lines does not trigger a paragraph break."""
    lines = [
        _line("Line one of para", 0, 20),
        _line("line two of para", 22, 42),   # gap = 2, tiny
    ]
    text = _clean(lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    assert len(paragraphs) == 1


# ---------------------------------------------------------------------------
# Mixed scenario
# ---------------------------------------------------------------------------

def test_mixed_scenario():
    """Multiple lines with hyphens, continuations, and block breaks."""
    lines = [
        # Paragraph 1: two continuation lines + dehyphenated word
        _line("The govern-",    0,  20),
        _line("ment decided",  22,  42),
        _line("to act quickly.", 44, 64),
        # Gap → new paragraph
        _line("Meanwhile the",  130, 150),
        _line("crowd dispersed.", 152, 172),
    ]
    text = _clean(lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    assert len(paragraphs) == 2
    # First paragraph should contain the dehyphenated word.
    assert "government" in paragraphs[0]
    assert "quickly" in paragraphs[0]
    # Second paragraph.
    assert "crowd" in paragraphs[1]


# ---------------------------------------------------------------------------
# Disabled cleaner
# ---------------------------------------------------------------------------

def test_disabled_cleaner_passes_through():
    """When enabled=False, clean() returns layout unchanged."""
    page_img = Image.fromarray(np.zeros((100, 400, 3), dtype=np.uint8))
    lines = [_line("some-", 0, 20), _line("thing here", 22, 42)]
    region = Region(
        bbox=BBox(0, 0, 400, 100),
        image=page_img,
        label="article",
        lines=lines,
        text="original",
    )
    layout = PageLayout(image=page_img, regions=[region], width=400, height=100)
    cleaner = TextCleaner(enabled=False)
    result = cleaner.clean(layout)
    assert result.regions[0].text == "original"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_lines():
    """Regions with no lines return empty string."""
    page_img = Image.fromarray(np.zeros((100, 400, 3), dtype=np.uint8))
    region = Region(
        bbox=BBox(0, 0, 400, 100), image=page_img, label="article", lines=[]
    )
    layout = PageLayout(image=page_img, regions=[region], width=400, height=100)
    cleaner = TextCleaner(enabled=True)
    result = cleaner.clean(layout)
    assert result.regions[0].text == ""


def test_single_line():
    """A single line is returned as-is."""
    lines = [_line("Just one line.", 0, 20)]
    text = _clean(lines)
    assert text == "Just one line."


def test_blank_line_text_skipped():
    """Lines with empty text are skipped without crashing."""
    lines = [
        _line("First line", 0, 20),
        _line("",           22, 42),   # blank
        _line("third line", 44, 64),
    ]
    text = _clean(lines)
    assert "First line" in text
    assert "third line" in text
