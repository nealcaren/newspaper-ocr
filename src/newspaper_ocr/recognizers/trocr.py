from __future__ import annotations

from pathlib import Path

from PIL import Image

from newspaper_ocr.models import Line
from newspaper_ocr.recognizers.base import LineRecognizer

# Default HuggingFace model for the finetuned version
_HF_FINETUNED = "NealCaren/trocr-newspaper-ocr"
_HF_BASE = "microsoft/trocr-base-printed"


class TrOCRRecognizer(LineRecognizer):
    """TrOCR line-level OCR recognizer.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or local path.  Defaults to the finetuned
        newspaper model.  Use "microsoft/trocr-base-printed" for the
        stock model.
    device : str, optional
        "cuda", "mps", or "cpu".  Auto-detected if None.
    """

    def __init__(
        self,
        model_id: str = _HF_FINETUNED,
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError:
            raise ImportError(
                "TrOCR dependencies not installed. Install with:\n"
                "  pip install 'newspaper-ocr[trocr]'"
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self._torch = torch

        self._processor = TrOCRProcessor.from_pretrained(model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
        self._model.eval()

    def recognize(self, line: Line) -> Line:
        try:
            image = line.image.convert("RGB")
            pixel_values = self._processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(self.device)

            with self._torch.no_grad():
                generated_ids = self._model.generate(pixel_values, max_new_tokens=128)

            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            line.text = text
            line.confidence = 0.9 if text else 0.0
        except Exception:
            line.text = ""
            line.confidence = 0.0
        return line
