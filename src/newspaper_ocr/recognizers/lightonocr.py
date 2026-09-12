from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from newspaper_ocr import repetition
from newspaper_ocr.errors import is_timeout
from newspaper_ocr.models import Region
from newspaper_ocr.recognizers.base import RegionRecognizer


class LightOnOcrRecognizer(RegionRecognizer):
    """LightOnOCR-2-1B vision-language model for OCR.

    Achieves ~1% CER on historical newspaper text but requires GPU.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
    device : str
        "cuda", "mps", or "cpu".
    max_new_tokens : int
        Maximum tokens to generate per region.
    """

    def __init__(
        self,
        model_id: str = "lightonai/LightOnOCR-2-1B-base",
        device: str | None = None,
        max_new_tokens: int = 512,
        repetition_min_len: int = repetition.MIN_LEN,
        repetition_min_reps: int = repetition.MIN_REPS,
    ):
        try:
            import torch
            from transformers import (
                LightOnOcrForConditionalGeneration,
                LightOnOcrProcessor,
            )
        except ImportError:
            raise ImportError(
                "LightOnOCR dependencies not installed. Install with:\n"
                "  pip install 'newspaper-ocr[lightonocr]'"
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.repetition_min_len = repetition_min_len
        self.repetition_min_reps = repetition_min_reps
        self._torch = torch

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._processor = LightOnOcrProcessor.from_pretrained(model_id)
        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)

    # Shared with the other VLM recognizers; see newspaper_ocr.repetition.
    _has_repetition = staticmethod(repetition.has_repetition)
    _truncate_repetition = staticmethod(repetition.truncate_repetition)

    def recognize(self, region: Region) -> Region:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            region.image.save(f.name)
            tmp_path = f.name

        try:
            conversation = [
                {"role": "user", "content": [{"type": "image", "url": tmp_path}]}
            ]
            inputs = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

            with self._torch.no_grad():
                output = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )

            text = self._processor.decode(
                output[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            if repetition.has_repetition(
                text, self.repetition_min_len, self.repetition_min_reps
            ):
                region.text = repetition.truncate_repetition(
                    text, self.repetition_min_len
                )
                region.status = "repetition"
            else:
                region.text = text
                region.status = "ok"
        except Exception as exc:
            region.text = ""
            region.status = "timeout" if is_timeout(exc) else "error"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return region
