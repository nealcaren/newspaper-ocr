from __future__ import annotations
import tempfile
from pathlib import Path
from newspaper_ocr.models import Line
from newspaper_ocr.recognizers.base import LineRecognizer


class EffocrRecognizer(LineRecognizer):
    def __init__(self, model_dir: str | Path | None = None):
        try:
            from efficient_ocr import EffOCR
        except ImportError:
            raise ImportError(
                "efficient_ocr is not installed. Install with:\n"
                "  pip install 'newspaper-ocr[effocr]'\n"
                "  or: pip install git+https://github.com/nealcaren/efficient_ocr.git"
            )

        model_dir = str(model_dir) if model_dir else "./models"
        self.effocr = EffOCR(config={
            "Global": {"skip_line_detection": True},
            "Recognizer": {
                "char": {"model_backend": "onnx", "model_dir": model_dir,
                         "hf_repo_id": "dell-research-harvard/effocr_en/char_recognizer"},
                "word": {"model_backend": "onnx", "model_dir": model_dir,
                         "hf_repo_id": "dell-research-harvard/effocr_en/word_recognizer"},
            },
            "Localizer": {"model_backend": "onnx", "model_dir": model_dir,
                          "hf_repo_id": "dell-research-harvard/effocr_en"},
            "Line": {"model_backend": "onnx", "model_dir": model_dir,
                     "hf_repo_id": "dell-research-harvard/effocr_en"},
        })

    def recognize(self, line: Line) -> Line:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            line.image.save(f.name)
            try:
                results = self.effocr.infer(f.name)
                if results and results[0].text:
                    line.text = results[0].text.strip()
                    line.confidence = 0.8
            except Exception:
                line.text = ""
                line.confidence = 0.0
            finally:
                Path(f.name).unlink(missing_ok=True)
        return line
