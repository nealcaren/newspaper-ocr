from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
from newspaper_ocr.models import Line, Region
from newspaper_ocr.recognizers.base import LineRecognizer


class TesseractRecognizer(LineRecognizer):
    def __init__(
        self,
        model: str = "eng",
        tessdata_dir: str | Path | None = None,
        mode: str = "region",
    ):
        """
        Tesseract OCR recognizer using subprocess calls.

        Parameters
        ----------
        model : str
            Tesseract language/model name (e.g. "eng").
        tessdata_dir : str or Path, optional
            Custom tessdata directory.
        mode : str
            "line" (psm 7, one call per line) or "region" (psm 6, one call per region).
        """
        if mode not in ("line", "region"):
            raise ValueError(f"mode must be 'line' or 'region', got '{mode}'")
        self.model = model
        self.tessdata_dir = str(tessdata_dir) if tessdata_dir else None
        self.mode = mode
        self._check_installed()

    def _check_installed(self):
        try:
            subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        except FileNotFoundError:
            raise ImportError(
                "Tesseract is not installed. Install with:\n"
                "  macOS: brew install tesseract\n"
                "  Ubuntu: apt install tesseract-ocr"
            )

    def _build_cmd(self, image_path: str, psm: str) -> list[str]:
        cmd = ["tesseract", image_path, "stdout", "--psm", psm]
        if self.tessdata_dir:
            cmd.extend(["--tessdata-dir", self.tessdata_dir])
        cmd.extend(["-l", self.model])
        return cmd

    def recognize(self, line: Line) -> Line:
        """Recognize text in a single line crop (psm 7)."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            line.image.save(f.name)
            cmd = self._build_cmd(f.name, "7")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            Path(f.name).unlink(missing_ok=True)
        line.text = result.stdout.strip()
        line.confidence = 0.9 if line.text else 0.0
        return line

    def recognize_region(self, region: Region) -> Region:
        """Recognize all text in a region crop using Tesseract's internal line segmentation (psm 6)."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            region.image.save(f.name)
            cmd = self._build_cmd(f.name, "6")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            Path(f.name).unlink(missing_ok=True)
        region.text = result.stdout.strip()
        region.confidence = 0.9 if region.text else 0.0
        return region
