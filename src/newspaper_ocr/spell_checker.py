"""Optional SymSpell-based spell correction for OCR output."""
from __future__ import annotations

import re
from typing import NamedTuple

from newspaper_ocr.models import PageLayout


class Correction(NamedTuple):
    region_index: int
    original: str
    corrected: str


# Regex to split leading/trailing punctuation from a word token.
_LEADING_PUNCT = re.compile(r'^([^a-zA-Z0-9]*)(.*?)([^a-zA-Z0-9]*)$', re.DOTALL)

# Words that look like numbers (digits only, or things like "1856").
_NUMBER_RE = re.compile(r'^\d+$')

# Mixed-case words like McClellan, iPhone — skip these.
# Pattern: a lowercase letter followed by an uppercase (e.g. cC in McClellan).
_MIXED_CASE_RE = re.compile(r'[a-z][A-Z]')


class SpellChecker:
    """Optional SymSpell-based spell correction for OCR output."""

    def __init__(
        self,
        enabled: bool = True,
        max_edit_distance: int = 2,
        dictionary_path: str | None = None,
    ):
        """
        Args:
            enabled: Whether to run spell correction.
            max_edit_distance: Max edit distance for corrections (1 or 2).
            dictionary_path: Path to custom frequency dictionary.
                             If None, uses symspellpy's built-in English dictionary.
        """
        self.enabled = enabled
        self.corrections: list[Correction] = []
        self._sym_spell = None

        if not enabled:
            return

        from symspellpy import SymSpell, Verbosity  # type: ignore

        self._verbosity = Verbosity.CLOSEST
        self._sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance)

        if dictionary_path is not None:
            self._sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        else:
            import pathlib
            import symspellpy as _symspellpy_pkg
            dict_path = str(
                pathlib.Path(_symspellpy_pkg.__file__).parent
                / "frequency_dictionary_en_82_765.txt"
            )
            self._sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

    def check(self, layout: PageLayout) -> PageLayout:
        """Run spell correction on all region text."""
        if not self.enabled or self._sym_spell is None:
            return layout

        self.corrections = []

        for region_idx, region in enumerate(layout.regions):
            if not region.text:
                continue
            region.text = self._correct_text(region.text, region_idx)

        return layout

    def _correct_text(self, text: str, region_idx: int) -> str:
        """Correct all words in a block of text, preserving structure."""
        words = text.split(" ")
        corrected_words = []
        for word in words:
            corrected = self._correct_token(word, region_idx)
            corrected_words.append(corrected)
        return " ".join(corrected_words)

    def _correct_token(self, token: str, region_idx: int) -> str:
        """Correct a single whitespace-separated token (may include punctuation)."""
        if not token:
            return token

        # Split off leading and trailing punctuation.
        m = _LEADING_PUNCT.match(token)
        if m is None:
            return token
        lead, word, trail = m.group(1), m.group(2), m.group(3)

        corrected_word = self._correct_word(word)

        if corrected_word != word:
            original_token = token
            corrected_token = lead + corrected_word + trail
            self.corrections.append(
                Correction(
                    region_index=region_idx,
                    original=original_token,
                    corrected=corrected_token,
                )
            )
            return corrected_token

        return token

    def _correct_word(self, word: str) -> str:
        """Correct a single word if it's not in the dictionary."""
        if not word:
            return word

        # Skip very short words — too many false positives.
        if len(word) <= 2:
            return word

        # Skip pure numbers.
        if _NUMBER_RE.match(word):
            return word

        # Skip mixed-case words like McClellan, iPhone (lowercase then uppercase).
        if _MIXED_CASE_RE.search(word):
            return word

        lower_word = word.lower()

        # Check if the word is already in the dictionary (no correction needed).
        # lookup returns results even for known words; if the top result has
        # edit_distance == 0, the word is known.
        results = self._sym_spell.lookup(
            lower_word, self._verbosity, include_unknown=False
        )
        if not results:
            return word

        top = results[0]

        # Word is already correctly spelled — leave it alone.
        if top.distance == 0:
            return word

        # Word is not in the dictionary — apply correction with case preservation.
        return self._preserve_case(word, top.term)

    def _preserve_case(self, original: str, correction: str) -> str:
        """Apply original's capitalisation pattern to correction."""
        if not original or not correction:
            return correction

        # All-uppercase: "CARRISON" → "GARRISON"
        if original.isupper():
            return correction.upper()

        # Title-case (first letter upper, rest lower): "Carrison" → "Garrison"
        if original[0].isupper() and original[1:].islower():
            return correction.capitalize()

        # All-lowercase: "carrison" → "garrison"
        if original.islower():
            return correction.lower()

        # Fallback: return as-is from the dictionary.
        return correction
