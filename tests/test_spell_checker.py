"""Tests for SpellChecker post-processing stage."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from newspaper_ocr.models import BBox, PageLayout, Region
from newspaper_ocr.spell_checker import SpellChecker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layout(text: str) -> PageLayout:
    """Create a minimal PageLayout with a single region containing the given text."""
    img = Image.fromarray(np.zeros((100, 400, 3), dtype=np.uint8))
    region = Region(
        bbox=BBox(0, 0, 400, 100),
        image=img,
        label="article",
        text=text,
    )
    return PageLayout(image=img, regions=[region], width=400, height=100)


def _check(text: str, **kwargs) -> tuple[str, SpellChecker]:
    """Run SpellChecker on a single-region layout and return (corrected_text, checker)."""
    checker = SpellChecker(**kwargs)
    layout = _layout(text)
    result = checker.check(layout)
    return result.regions[0].text, checker


# ---------------------------------------------------------------------------
# Core correction behaviour
#
# We use "receeve" → "receive" as the canonical test word because SymSpell
# returns exactly one candidate (unambiguous, no competing words at distance 1).
# "Carrison" is also kept in the docstring as the motivating example, but
# SymSpell resolves it to "Harrison" (higher frequency) not "Garrison".
# ---------------------------------------------------------------------------

def test_corrects_obvious_ocr_error():
    """'receeve' → 'receive' (obvious OCR substitution, unambiguous candidate)."""
    text, _ = _check("receeve")
    assert text == "receive"


def test_preserves_known_words():
    """'the' stays 'the', not mutated to something else."""
    text, _ = _check("the")
    assert text == "the"


def test_preserves_known_word_in_sentence():
    """Common words in context are not altered."""
    text, _ = _check("the soldiers marched")
    assert "the" in text
    assert "soldiers" in text
    assert "marched" in text


# ---------------------------------------------------------------------------
# Capitalisation preservation
# ---------------------------------------------------------------------------

def test_preserves_capitalization_upper():
    """'RECEEVE' → 'RECEIVE' (all-caps preserved)."""
    text, _ = _check("RECEEVE")
    assert text == "RECEIVE"


def test_preserves_capitalization_lower():
    """'receeve' → 'receive' (all-lowercase preserved)."""
    text, _ = _check("receeve")
    assert text == "receive"


def test_preserves_capitalization_title():
    """'Receeve' → 'Receive' (title-case preserved)."""
    text, _ = _check("Receeve")
    assert text == "Receive"


# ---------------------------------------------------------------------------
# Punctuation handling
# ---------------------------------------------------------------------------

def test_preserves_punctuation_trailing_comma():
    """'Receeve,' → 'Receive,' (trailing comma reattached)."""
    text, _ = _check("Receeve,")
    assert text == "Receive,"


def test_preserves_punctuation_trailing_period():
    """'Receeve.' → 'Receive.' (trailing period reattached)."""
    text, _ = _check("Receeve.")
    assert text == "Receive."


def test_preserves_punctuation_leading():
    """Leading punctuation is preserved and reattached."""
    text, _ = _check('"Receeve"')
    assert text == '"Receive"'


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def test_skips_short_words():
    """Words with 2 or fewer characters are not corrected."""
    text, checker = _check("an")
    assert text == "an"
    assert len(checker.corrections) == 0


def test_skips_single_char():
    """Single-character words (like 'I' or 'a') are left alone."""
    text, checker = _check("I")
    assert text == "I"
    assert len(checker.corrections) == 0


def test_skips_numbers():
    """Pure numeric tokens like '1856' are not corrected."""
    text, checker = _check("1856")
    assert text == "1856"
    assert len(checker.corrections) == 0


def test_skips_mixed_case_abbreviations():
    """Mixed-case proper nouns like 'McClellan' are not corrected."""
    text, checker = _check("McClellan")
    assert text == "McClellan"
    assert len(checker.corrections) == 0


# ---------------------------------------------------------------------------
# Disabled checker
# ---------------------------------------------------------------------------

def test_disabled():
    """When enabled=False, text passes through completely unchanged."""
    original = "receeve teh soldir"
    text, checker = _check(original, enabled=False)
    assert text == original
    assert len(checker.corrections) == 0


def test_disabled_check_returns_layout_unchanged():
    """When enabled=False, check() returns the same layout object."""
    checker = SpellChecker(enabled=False)
    layout = _layout("receeve")
    result = checker.check(layout)
    assert result.regions[0].text == "receeve"


# ---------------------------------------------------------------------------
# Corrections log
# ---------------------------------------------------------------------------

def test_corrections_logged():
    """spell_checker.corrections tracks every word that was changed."""
    _, checker = _check("receeve")
    assert len(checker.corrections) >= 1
    original_words = [c.original for c in checker.corrections]
    assert "receeve" in original_words


def test_corrections_log_includes_region_index():
    """Each logged correction records which region it came from."""
    _, checker = _check("receeve")
    assert all(c.region_index == 0 for c in checker.corrections)


def test_corrections_log_cleared_on_each_check():
    """Running check() twice on fresh layouts resets the corrections list."""
    checker = SpellChecker()
    # First run on a misspelled word.
    layout1 = _layout("receeve")
    checker.check(layout1)
    first_count = len(checker.corrections)
    assert first_count >= 1

    # Second run: corrections list is reset before processing.
    layout2 = _layout("receive")   # already correct — no new corrections
    checker.check(layout2)
    second_count = len(checker.corrections)
    assert second_count == 0   # reset, not accumulated


def test_known_words_not_logged():
    """Correctly spelled words do not appear in the corrections log."""
    _, checker = _check("receive")
    corrected_originals = [c.original for c in checker.corrections]
    assert "receive" not in corrected_originals


# ---------------------------------------------------------------------------
# Custom dictionary
# ---------------------------------------------------------------------------

def test_custom_dictionary(tmp_path):
    """Can load a custom frequency dictionary file."""
    dict_file = tmp_path / "custom.txt"
    dict_file.write_text("zymble 1000\n")

    checker = SpellChecker(dictionary_path=str(dict_file))
    # With only 'zymble' in the dict, 'zymble' has distance 0 and is left alone.
    layout = _layout("zymble")
    result = checker.check(layout)
    assert result.regions[0].text == "zymble"
    assert len(checker.corrections) == 0


# ---------------------------------------------------------------------------
# Multi-word and multi-region edge cases
# ---------------------------------------------------------------------------

def test_corrects_word_in_sentence():
    """Correction works when the misspelled word is embedded in a sentence."""
    text, _ = _check("Please receeve the letter.")
    assert "receive" in text
    assert "Please" in text
    assert "letter." in text


def test_multiple_corrections_in_one_region():
    """Multiple misspelled words in one region are all corrected."""
    text, checker = _check("receeve RECEEVE Receeve")
    assert "receive" in text
    assert "RECEIVE" in text
    assert "Receive" in text
    assert len(checker.corrections) == 3


def test_empty_region_text():
    """Regions with empty text don't crash."""
    checker = SpellChecker()
    layout = _layout("")
    result = checker.check(layout)
    assert result.regions[0].text == ""
    assert len(checker.corrections) == 0
