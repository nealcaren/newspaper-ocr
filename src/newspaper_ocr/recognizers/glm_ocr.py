from __future__ import annotations

import contextlib
import signal
import threading
import time

from PIL import Image

from newspaper_ocr import repetition
from newspaper_ocr.errors import OcrTimeout, is_timeout
from newspaper_ocr.models import Region
from newspaper_ocr.recognizers.base import RegionRecognizer

#: Placeholder text written when a region exhausts its retries on a timeout.
#: Matches the production pipeline so downstream reports can grep for it.
TIMEOUT_TEXT = "[OCR timeout]"


@contextlib.contextmanager
def _wall_clock_alarm(seconds: float):
    """Raise :class:`OcrTimeout` if the wrapped block runs longer than *seconds*.

    Uses ``SIGALRM``, which interrupts even a single blocking forward pass, but
    is only available on Unix from the main thread.  Yields True when the alarm
    is armed, and False when it could not be — in which case the caller's
    between-token deadline is the only guard, and a call hung inside a single
    forward pass will run to completion before being reported as a timeout.
    """
    armed = (
        seconds
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not armed:
        yield False
        return

    def _on_alarm(signum, frame):
        raise OcrTimeout(f"OCR exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class GlmOcrRecognizer(RegionRecognizer):
    """GLM-OCR vision-language model. Requires GPU or MLX server.

    Two modes:
    - mode="api": Connect to running MLX/vLLM server (default)
    - mode="local": Load model directly via transformers (GPU required)

    ``timeout`` is a per-region wall-clock budget and applies in both modes: it
    configures the HTTP client in API mode and guards ``generate()`` in local
    mode, so a pathological region can't hang a whole batch.  A region that
    exhausts its retries is left with :data:`TIMEOUT_TEXT` and
    ``status="timeout"`` rather than silently empty text.

    The strength of that guard depends on where it runs.  On the main thread of
    a Unix process ``SIGALRM`` interrupts ``generate()`` mid-call, so even a
    genuinely hung forward pass is cut off at the budget.  Off the main thread
    (or on Windows) the alarm can't be armed and only the between-token deadline
    applies: generation still stops early, but a call that hangs *inside* one
    forward pass is only reported as a timeout once it returns.  Run batches on
    the main thread if you need hangs bounded rather than merely detected.

    ``repetition_min_len`` / ``repetition_min_reps`` tune the loop detector; the
    defaults match the production pipeline (tag ``2025-03-07-col-fix``).
    """

    def __init__(
        self,
        mode: str = "api",
        api_url: str = "http://localhost:8080/v1/chat/completions",
        model_id: str = "zai-org/GLM-OCR",
        mlx_model_id: str = "mlx-community/GLM-OCR-bf16",
        timeout: float = 120,
        max_retries: int = 2,
        repetition_min_len: int = repetition.MIN_LEN,
        repetition_min_reps: int = repetition.MIN_REPS,
    ):
        self.mode = mode
        self.api_url = api_url
        self.model_id = model_id
        self.mlx_model_id = mlx_model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.repetition_min_len = repetition_min_len
        self.repetition_min_reps = repetition_min_reps

        # Lazy-loaded for local mode
        self._model = None
        self._processor = None

        # Lazy-loaded for API mode
        self._client = None

        if mode == "api":
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx is not installed. Install with:\n"
                    "  pip install httpx"
                )
            self._client = httpx.Client(timeout=timeout)
        elif mode == "local":
            # Validate that transformers is available but don't load model yet
            try:
                import transformers  # noqa: F401
            except ImportError:
                raise ImportError(
                    "transformers is not installed. Install with:\n"
                    "  pip install 'transformers>=5.1' torch\n"
                    "Note: GLM-OCR requires transformers>=5.1"
                )
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")

    def _recognize_api(self, image: Image.Image) -> str:
        """Call MLX/vLLM server. Returns text."""
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = self._client.post(
            self.api_url,
            json={
                "model": self.mlx_model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}"
                                },
                            },
                            {"type": "text", "text": "Text Recognition:"},
                        ],
                    }
                ],
                "max_tokens": 4096,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _load_model(self):
        """Lazy-load the transformers model."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers is not installed. Install with:\n"
                "  pip install 'transformers>=5.1' torch\n"
                "Note: GLM-OCR requires transformers>=5.1"
            )
        import torch

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        # MPS has known issues with GLM-OCR vision position ids;
        # use CUDA when available, otherwise fall back to CPU.
        #
        # Load onto the device with an explicit ``.to(...)`` rather than
        # ``device_map="auto"``: the latter needs the optional ``accelerate``
        # package, and without it ``from_pretrained`` raises — which the per-region
        # try/except in :meth:`recognize` swallows into ``status="error"`` with
        # empty text, so every region silently comes back blank. These models are
        # small (single-GPU), so a plain ``.to("cuda")`` is correct and dependency-free.
        if torch.cuda.is_available():
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=torch.bfloat16
            ).to("cuda")
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=torch.float32
            )

    def _recognize_local(self, image: Image.Image) -> str:
        """Run model directly. Returns text."""
        import base64
        import io

        import torch

        self._load_model()

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{b64}"},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": 4096}
        deadline = time.monotonic() + self.timeout if self.timeout else None
        if deadline is not None:
            gen_kwargs["stopping_criteria"] = self._deadline_criteria(deadline)

        with torch.no_grad(), _wall_clock_alarm(self.timeout):
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # The stopping criteria halts between tokens, so generate() can return
        # normally after the budget is spent; treat that as a timeout too.
        if deadline is not None and time.monotonic() >= deadline:
            raise OcrTimeout(f"OCR exceeded {self.timeout}s")

        return self._processor.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _deadline_criteria(deadline: float):
        """Stop generation once *deadline* passes.

        Backstop for platforms where ``SIGALRM`` isn't available (Windows, worker
        threads).  It only fires between tokens, so it complements rather than
        replaces the alarm.
        """
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return time.monotonic() >= deadline

        return StoppingCriteriaList([_Deadline()])

    # Aliases onto the shared implementation in newspaper_ocr.repetition, so
    # every VLM recognizer detects loops the same way.  Note the thresholds moved
    # to the production values (20 chars / 5 reps) when the algorithm was aligned.
    _has_repetition = staticmethod(repetition.has_repetition)
    _truncate_repetition = staticmethod(repetition.truncate_repetition)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove Markdown code-fence lines the model wraps transcriptions in.

        GLM-OCR sometimes returns the page text inside a ```` ```markdown ```` /
        ```` ``` ```` block — most often on small crops (so the residual pass
        hits it hardest).  The fence lines are non-transcription noise that
        corrupt line-based scoring, so drop any line that is only a code fence,
        keeping the transcribed content between them.
        """
        if "```" not in text:
            return text
        lines = [ln for ln in text.splitlines()
                 if not ln.lstrip().startswith("```")]
        return "\n".join(lines).strip()

    def recognize(self, region: Region) -> Region:
        for attempt in range(self.max_retries + 1):
            try:
                if self.mode == "api":
                    text = self._recognize_api(region.image)
                else:
                    text = self._recognize_local(region.image)
                text = self._strip_markdown_fences(text)
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
