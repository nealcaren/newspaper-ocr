from __future__ import annotations
from abc import ABC, abstractmethod
from newspaper_ocr.models import Line, Region


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
        """Recognize text in a full region crop."""
