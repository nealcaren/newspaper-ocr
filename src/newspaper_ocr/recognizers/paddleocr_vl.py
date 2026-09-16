from __future__ import annotations

import time

from PIL import Image

from newspaper_ocr import repetition
from newspaper_ocr.errors import OcrTimeout, is_timeout
from newspaper_ocr.models import Region
from newspaper_ocr.recognizers.base import RegionRecognizer
from newspaper_ocr.recognizers.glm_ocr import (
    TIMEOUT_TEXT,
    GlmOcrRecognizer,
    _wall_clock_alarm,
)


class PaddleOcrVlRecognizer(RegionRecognizer):
    """PaddleOCR-VL vision-language model for region-level OCR. GPU recommended.

    Loaded directly via transformers, which has native PaddleOCR-VL support from
    5.3 on — so it loads *without* ``trust_remote_code`` (the checkpoint's bundled
    remote code targets an older transformers and breaks on current versions).

    Behaves like :class:`GlmOcrRecognizer` in local mode: a per-region
    ``timeout`` guards ``generate()`` (``SIGALRM`` on the main thread, a
    between-token deadline elsewhere), and the shared repetition detector tags
    loops.  A region that exhausts its retries on a timeout is left with
    :data:`TIMEOUT_TEXT` and ``status="timeout"``.  This is the recognizer used
    to recover regions a primary model (e.g. GLM-OCR) failed on.

    ``repetition_min_len`` / ``repetition_min_reps`` tune the loop detector; the
    defaults match the other VLM recognizers.
    """

    def __init__(
        self,
        model_id: str = "PaddlePaddle/PaddleOCR-VL-1.6",
        prompt: str = "OCR:",
        device: str | None = None,
        timeout: float = 120,
        max_new_tokens: int = 1024,
        max_retries: int = 1,
        repetition_min_len: int = repetition.MIN_LEN,
        repetition_min_reps: int = repetition.MIN_REPS,
    ):
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "PaddleOCR-VL dependencies not installed. Install with:\n"
                "  pip install 'newspaper-ocr[paddleocr-vl]'\n"
                "Note: native PaddleOCR-VL support requires transformers>=5.3"
            )

        self.model_id = model_id
        self.prompt = prompt
        self.device = device
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.repetition_min_len = repetition_min_len
        self.repetition_min_reps = repetition_min_reps

        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load the transformers model and processor."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            raise ImportError(
                "PaddleOCR-VL dependencies not installed. Install with:\n"
                "  pip install 'newspaper-ocr[paddleocr-vl]'\n"
                "Note: native PaddleOCR-VL support requires transformers>=5.3"
            )

        import torch

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        # Load with dtype="auto" (checkpoint's native dtype), then place with an
        # explicit ``.to(...)`` rather than ``device_map="auto"``. device_map needs
        # the optional ``accelerate`` package; without it ``from_pretrained`` raises,
        # and the per-region try/except in :meth:`recognize` swallows that into
        # ``status="error"`` with empty text — every region silently blank. This
        # model is single-GPU-sized, so plain ``.to(device)`` is correct.
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype="auto"
        ).to(dev).eval()

    def _recognize_local(self, image: Image.Image) -> str:
        """Run the model on a region image. Returns recognized text."""
        import torch

        self._load_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs = {"max_new_tokens": self.max_new_tokens}
        deadline = time.monotonic() + self.timeout if self.timeout else None
        if deadline is not None:
            gen_kwargs["stopping_criteria"] = GlmOcrRecognizer._deadline_criteria(
                deadline
            )

        with torch.no_grad(), _wall_clock_alarm(self.timeout):
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # The stopping criteria halts between tokens, so generate() can return
        # normally after the budget is spent; treat that as a timeout too.
        if deadline is not None and time.monotonic() >= deadline:
            raise OcrTimeout(f"OCR exceeded {self.timeout}s")

        return self._processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0].strip()

    # Shared with the other VLM recognizers; see newspaper_ocr.repetition.
    _has_repetition = staticmethod(repetition.has_repetition)
    _truncate_repetition = staticmethod(repetition.truncate_repetition)

    def recognize(self, region: Region) -> Region:
        for attempt in range(self.max_retries + 1):
            try:
                text = self._recognize_local(region.image)
            except Exception as exc:
                if attempt < self.max_retries:
                    continue
                timed_out = is_timeout(exc)
                region.text = TIMEOUT_TEXT if timed_out else ""
                region.status = "timeout" if timed_out else "error"
                return region

            if not repetition.has_repetition(
                text, self.repetition_min_len, self.repetition_min_reps
            ):
                region.text = text
                region.status = "ok"
                return region
            if attempt < self.max_retries:
                continue
            region.text = repetition.truncate_repetition(
                text, self.repetition_min_len
            )
            region.status = "repetition"
            return region
        return region
