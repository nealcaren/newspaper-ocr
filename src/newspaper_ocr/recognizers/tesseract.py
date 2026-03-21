from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
from newspaper_ocr.models import Line
from newspaper_ocr.recognizers.base import LineRecognizer


class TesseractRecognizer(LineRecognizer):
    def __init__(self, model: str = "eng", tessdata_dir: str | Path | None = None):
        self.model = model
        self.tessdata_dir = str(tessdata_dir) if tessdata_dir else None
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

    def recognize(self, line: Line) -> Line:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            line.image.save(f.name)
            cmd = ["tesseract", f.name, "stdout", "--psm", "7"]
            if self.tessdata_dir:
                cmd.extend(["--tessdata-dir", self.tessdata_dir])
            cmd.extend(["-l", self.model])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            Path(f.name).unlink(missing_ok=True)
        line.text = result.stdout.strip()
        line.confidence = 0.9 if line.text else 0.0
        return line
