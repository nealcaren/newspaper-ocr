"""Core data models for the OCR pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Line:
    bbox: BBox
    image: Image.Image
    text: str = ""
    confidence: float = 0.0


@dataclass
class Region:
    bbox: BBox
    image: Image.Image
    label: str
    lines: list[Line] = field(default_factory=list)
    text: str = ""
    confidence: float = 0.0


@dataclass
class PageLayout:
    image: Image.Image
    regions: list[Region] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(r.text for r in self.regions if r.text)
