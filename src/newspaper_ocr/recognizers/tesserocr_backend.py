from __future__ import annotations
import os
import subprocess
from pathlib import Path
from newspaper_ocr.models import Line, Region
from newspaper_ocr.recognizers.base import LineRecognizer

try:
    import tesserocr
except ImportError:
    tesserocr = None


def _find_tessdata() -> str:
    """Discover the tessdata directory from environment or tesseract binary."""
    # 1. Environment variable
    env_path = os.environ.get("TESSDATA_PREFIX")
    if env_path and Path(env_path).is_dir():
        return env_path

    # 2. Ask the tesseract binary for its prefix
    try:
        result = subprocess.run(
            ["tesseract", "--print-parameters"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stderr.splitlines() + result.stdout.splitlines():
            if "tessdata" in line.lower() and "/" in line:
                candidate = line.split()[-1]
                if Path(candidate).is_dir():
                    return candidate
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Common Homebrew / system paths
    for candidate in [
        "/opt/homebrew/share/tessdata",
        "/usr/local/share/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tessdata",
    ]:
        if Path(candidate).is_dir():
            return candidate

    # 4. Fall back to tesserocr's own report (may be wrong)
    if tesserocr is not None:
        default = tesserocr.get_languages()[0]
        if default and Path(default).is_dir():
            return default

    raise RuntimeError(
        "Cannot find tessdata directory. Set TESSDATA_PREFIX or pass tessdata_dir."
    )


class TesserocrRecognizer(LineRecognizer):
    """Fast Tesseract recognition via C API bindings (no subprocess overhead)."""

    def __init__(
        self,
        model: str = "eng",
        tessdata_dir: str | Path | None = None,
        mode: str = "line",
    ):
        if tesserocr is None:
            raise ImportError(
                "tesserocr is not installed. Install with:\n"
                "  pip install tesserocr\n"
                "(requires libtesseract-dev system library)"
            )
        if mode not in ("line", "region"):
            raise ValueError(f"mode must be 'line' or 'region', got '{mode}'")
        self.mode = mode
        self.model = model

        tessdata = str(tessdata_dir) if tessdata_dir else _find_tessdata()
        self.api = tesserocr.PyTessBaseAPI(path=tessdata, lang=model)
        if mode == "region":
            self.api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
        else:
            self.api.SetPageSegMode(tesserocr.PSM.SINGLE_LINE)

    def recognize(self, line: Line) -> Line:
        """Recognize text in a single line crop via the C API."""
        self.api.SetImage(line.image)
        line.text = self.api.GetUTF8Text().strip()
        line.confidence = self.api.MeanTextConf() / 100.0
        return line

    def recognize_region(self, region: Region) -> Region:
        """Recognize all text in a region crop via the C API."""
        self.api.SetImage(region.image)
        region.text = self.api.GetUTF8Text().strip()
        region.confidence = self.api.MeanTextConf() / 100.0
        return region

    def __del__(self):
        if hasattr(self, "api"):
            self.api.End()
