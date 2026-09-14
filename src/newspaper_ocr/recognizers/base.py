from __future__ import annotations
from abc import ABC, abstractmethod

from PIL import Image

from newspaper_ocr.models import BBox, Line, Region


class LineRecognizer(ABC):
    @abstractmethod
    def recognize(self, line: Line) -> Line:
        """Recognize text in a single line crop."""

    def recognize_batch(self, lines: list[Line]) -> list[Line]:
        """Recognize text in multiple lines. Override for efficiency."""
        return [self.recognize(line) for line in lines]


class RegionRecognizer(ABC):
    @abstractmethod
    def recognize(self, region: Region) -> Region:
        """Recognize text in a full region crop and return the region.

        Return contract — every implementation must honour it, and every caller
        may rely on it:

        * The return value is a :class:`~newspaper_ocr.models.Region`, not a
          ``(text, status)`` tuple and not a bare string.  Implementations may
          mutate *region* in place and return it, which is what the bundled
          recognizers do.
        * ``region.text`` is a ``str`` — ``""`` when nothing was recognized,
          never ``None`` and never a container.
        * ``region.status`` is one of
          :data:`~newspaper_ocr.models.REGION_STATUSES`.

        The ``str`` half of that matters more than it looks.  A recognizer that
        returns ``(text, status)`` and a caller that assigns the whole tuple to
        ``region.text`` produce JSON with list-typed ``"text"``, which type
        checks nowhere and breaks every downstream consumer that expects a
        string — quietly, one page at a time.  Callers that recognize a bare
        crop (a re-OCR pass, a repair stage) should go through
        :func:`recognize_crop`, which normalizes any of those shapes back to
        ``(str, str)``.
        """


def recognize_crop(
    recognizer: RegionRecognizer | object,
    image: Image.Image,
    label: str = "text",
) -> tuple[str, str]:
    """Recognize a bare image crop, always returning ``(text, status)`` strings.

    For passes that re-OCR a piece of a page — a split container strip, a merged
    ad — rather than a detected region: it wraps *image* in a throwaway
    :class:`~newspaper_ocr.models.Region`, calls the recognizer, and normalizes
    whatever comes back.

    Accepts any region-capable recognizer: one with ``recognize_region`` (the
    line recognizers' region fallback) or a
    :class:`RegionRecognizer`.  It also absorbs the shapes the contract above
    rules out — a returned ``(text, status)`` tuple, a bare string, or ``None``
    from an implementation that only mutates in place — so a caller can never
    end up assigning a non-string to ``Region.text``.

    Raises
    ------
    TypeError
        If *recognizer* cannot recognize a region at all (a line-only
        recognizer with no ``recognize_region``).
    """
    region = Region(
        bbox=BBox(0, 0, image.width, image.height),
        image=image,
        label=label,
    )

    if hasattr(recognizer, "recognize_region"):
        result = recognizer.recognize_region(region)
    elif isinstance(recognizer, LineRecognizer):
        # Its recognize() takes a Line, so calling it with a Region would not
        # fail — it would quietly return text for the wrong kind of input.
        raise TypeError(
            f"{type(recognizer).__name__} is a line recognizer with no "
            "recognize_region(); it cannot read a region crop."
        )
    elif hasattr(recognizer, "recognize"):
        result = recognizer.recognize(region)
    else:
        raise TypeError(
            f"{type(recognizer).__name__} cannot recognize a region crop: "
            "it has neither recognize_region() nor recognize()."
        )

    return _as_text_status(result, region)


def _as_text_status(result: object, region: Region) -> tuple[str, str]:
    """Normalize a recognizer's return value to ``(text, status)`` strings."""
    if result is None:
        # In-place mutation only — read it back off the region we passed in.
        result = region

    if isinstance(result, Region):
        text, status = result.text, result.status
    elif isinstance(result, str):
        text, status = result, "ok"
    elif isinstance(result, (tuple, list)):
        # An off-contract (text, status) return. Take the text and, when it is
        # there, the status; anything longer is still only those two fields.
        text = result[0] if result else ""
        status = result[1] if len(result) > 1 else "ok"
    else:
        text, status = result, "ok"

    if text is None:
        text = ""
    if not isinstance(text, str):
        # Nested containers (a tuple holding a tuple) have no sane string form;
        # refuse rather than poison the page with str(("...", "ok")).
        raise TypeError(
            f"recognizer returned non-text {type(text).__name__} for a crop; "
            "RegionRecognizer.recognize must yield str text"
        )
    if not isinstance(status, str) or not status:
        status = "ok"
    return text, status
