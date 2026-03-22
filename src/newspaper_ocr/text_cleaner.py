"""Smart text reconstruction: join OCR'd lines into coherent paragraphs."""
from __future__ import annotations

import statistics

from newspaper_ocr.models import Line, PageLayout

# Words longer than this that end a line are treated as potential compounds.
_MIN_SEMANTIC_WORD_LEN = 3

# Punctuation that signals a sentence/block end.
_TERMINAL_PUNCT = set(".!?:;")

# Ratio of gap to median line height that triggers a paragraph break.
_GAP_RATIO = 1.5

# Fraction of page width; if x-origin shifts by more than this it's a new column.
_COLUMN_SHIFT_RATIO = 0.15


class TextCleaner:
    """Reconstruct continuous text from OCR'd lines within regions."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def clean(self, layout: PageLayout) -> PageLayout:
        """Process all regions, joining lines into coherent text."""
        if not self.enabled:
            return layout
        for region in layout.regions:
            # Only reconstruct from lines if lines have text (line-level recognizer).
            # If region.text is already set (region-level recognizer like GLM-OCR),
            # don't overwrite it with empty line reconstructions.
            if region.lines and any(line.text for line in region.lines):
                region.text = self._reconstruct_region(region.lines)
        return layout

    def _reconstruct_region(self, lines: list[Line]) -> str:
        """Join lines into paragraphs using dehyphenation and continuation rules."""
        if not lines:
            return ""

        median_height = self._median_line_height(lines)
        page_width = max(line.bbox.x1 for line in lines)

        paragraphs: list[str] = []
        current_parts: list[str] = []

        for i, line in enumerate(lines):
            text = line.text.strip()
            if not text:
                continue

            if not current_parts:
                current_parts.append(text)
                continue

            prev_line = lines[i - 1]
            prev_text = current_parts[-1] if current_parts else ""

            # Determine how to join this line to the accumulated paragraph text.
            if self._is_block_break(prev_line, line, median_height, page_width):
                # Flush the current paragraph and start fresh.
                paragraphs.append(" ".join(current_parts))
                current_parts = [text]
            elif self._should_dehyphenate(prev_text, text):
                # Remove trailing hyphen and merge without space.
                current_parts[-1] = prev_text.rstrip().rstrip("-")
                current_parts.append(text)
                # Combine them now so future logic sees the merged word.
                merged = "".join(current_parts[-2:])
                current_parts = current_parts[:-2] + [merged]
            elif self._is_continuation(prev_text, text):
                current_parts.append(text)
            else:
                # Lines don't continue — flush and start new paragraph.
                paragraphs.append(" ".join(current_parts))
                current_parts = [text]

        if current_parts:
            paragraphs.append(" ".join(current_parts))

        return "\n\n".join(paragraphs)

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def _should_dehyphenate(self, current_text: str, next_text: str) -> bool:
        """Check if line-ending hyphen should be removed and words joined."""
        stripped = current_text.rstrip()
        if not stripped.endswith("-"):
            return False
        if self._is_semantic_dash(stripped):
            return False
        # Next line must start with a lowercase letter for typographic dehyphenation.
        if not next_text or not next_text[0].islower():
            return False
        return True

    def _is_semantic_dash(self, text: str) -> bool:
        """Check if trailing dash is semantic (em-dash, spaced dash) not typographic.

        Typographic hyphens appear at line ends where a word is split across lines
        and the continuation starts with lowercase.  Semantic dashes are em-dashes
        (—, --) or explicitly spaced dashes (" -") used for parenthetical asides.

        Note: we intentionally do NOT treat every word->hyphen as semantic because
        word fragments such as "impor-" are indistinguishable from "well-" by
        length alone, and erroneously suppressing dehyphenation is the bigger harm.
        """
        stripped = text.rstrip()

        # Em-dash patterns.
        if stripped.endswith("\u2014") or stripped.endswith("--"):
            return True

        # Explicit spaced dash: " -" or " --" at end (space before the dash).
        if stripped.endswith(" -") or stripped.endswith(" --"):
            return True

        return False

    def _is_block_break(
        self,
        current_line: Line,
        next_line: Line,
        median_height: float,
        page_width: int,
    ) -> bool:
        """Check if there should be a paragraph break between these lines."""
        # Large vertical gap.
        gap = next_line.bbox.y0 - current_line.bbox.y1
        if median_height > 0 and gap > _GAP_RATIO * median_height:
            return True

        # Column shift: x-origin moves significantly.
        shift = abs(next_line.bbox.x0 - current_line.bbox.x0)
        if page_width > 0 and shift > _COLUMN_SHIFT_RATIO * page_width:
            return True

        # Terminal punctuation on previous line + next line starts uppercase.
        prev_text = current_line.text.rstrip()
        if prev_text and prev_text[-1] in _TERMINAL_PUNCT:
            next_stripped = next_line.text.lstrip()
            if next_stripped and next_stripped[0].isupper():
                return True

        return False

    def _is_continuation(self, current_text: str, next_text: str) -> bool:
        """Check if next line continues the current sentence."""
        stripped = current_text.rstrip()
        if not stripped or not next_text:
            return False

        # If current line ends with terminal punctuation, don't continue.
        if stripped[-1] in _TERMINAL_PUNCT:
            return False

        # Next line starts with lowercase → clear continuation.
        if next_text[0].islower():
            return True

        # Next line starts with a digit (e.g., numbered list continuation) → join.
        if next_text[0].isdigit():
            return True

        return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _median_line_height(lines: list[Line]) -> float:
        heights = [line.bbox.height for line in lines if line.bbox.height > 0]
        if not heights:
            return 0.0
        return statistics.median(heights)
