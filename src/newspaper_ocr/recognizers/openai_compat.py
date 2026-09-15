"""Region recognizer for any OpenAI-compatible chat-completions endpoint.

This is the "bring your own OCR device" backend.  It talks the OpenAI vision
chat-completions protocol, so it works unchanged against OpenAI, OpenRouter,
Together, Groq, a local vLLM/LM-Studio/Ollama server, or anything else that
speaks that API — you just point ``base_url`` at it and name a ``model``.

Unlike the built-in VLM backends (glm-ocr, lightonocr, paddleocr-vl), nothing
is downloaded or run locally: the model lives behind the endpoint, so this stays
useful as hosted models come and go.

Example
-------
>>> from newspaper_ocr import Pipeline
>>> from newspaper_ocr.recognizers.openai_compat import OpenAiCompatRecognizer
>>> rec = OpenAiCompatRecognizer(
...     base_url="https://openrouter.ai/api/v1",
...     model="openai/gpt-4o-mini",
...     api_key_env="OPENROUTER_API_KEY",
... )
>>> Pipeline(recognizer=rec, output="text").run("page.jp2")

The ``openai`` and ``openrouter`` registry presets wire up the two most common
endpoints; supply the model via ``--model`` on the CLI or ``recognizer_model=``
in :class:`~newspaper_ocr.Pipeline`.
"""
from __future__ import annotations

import base64
import io
import os

from PIL import Image

from newspaper_ocr import repetition
from newspaper_ocr.errors import is_timeout
from newspaper_ocr.models import Region
from newspaper_ocr.recognizers.base import RegionRecognizer

#: Placeholder text written when a region exhausts its retries on a timeout.
#: Matches the other VLM backends so downstream reports can grep for it.
TIMEOUT_TEXT = "[OCR timeout]"

#: Default instruction sent alongside each region crop.
DEFAULT_PROMPT = (
    "Transcribe all text in this image exactly as it appears, preserving line "
    "breaks and reading order. Return only the transcribed text with no "
    "commentary, labels, or markdown fences."
)


class OpenAiCompatRecognizer(RegionRecognizer):
    """Recognize regions via an OpenAI-compatible ``/chat/completions`` endpoint.

    Parameters
    ----------
    model:
        Model identifier passed through to the endpoint (e.g. ``"gpt-4o-mini"``
        for OpenAI, ``"openai/gpt-4o-mini"`` for OpenRouter).
    base_url:
        API root **without** the trailing ``/chat/completions`` (that path is
        appended). Defaults to OpenAI's public endpoint.
    api_key:
        Bearer token. If ``None`` (the default), it is read from the environment
        variable named by ``api_key_env`` at construction time, so keys never
        have to be hard-coded.
    api_key_env:
        Name of the environment variable to read the key from when ``api_key``
        is not given. Defaults to ``"OPENAI_API_KEY"``.
    prompt:
        Instruction sent with each image. Override to tune for a language or
        document type.
    timeout:
        Per-region wall-clock budget (seconds) handed to the HTTP client. A
        region that exhausts its retries on a timeout is left with
        :data:`TIMEOUT_TEXT` and ``status="timeout"`` rather than empty text.
    max_retries:
        Extra attempts after the first on error, timeout, or a looping response.
    max_tokens:
        Upper bound on generated tokens per region.
    extra_headers:
        Additional HTTP headers merged into every request — handy for
        OpenRouter's optional ``HTTP-Referer`` / ``X-Title`` attribution.
    repetition_min_len / repetition_min_reps:
        Loop-detector thresholds, shared with the other VLM backends.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        prompt: str = DEFAULT_PROMPT,
        timeout: float = 60,
        max_retries: int = 2,
        max_tokens: int = 4096,
        extra_headers: dict[str, str] | None = None,
        repetition_min_len: int = repetition.MIN_LEN,
        repetition_min_reps: int = repetition.MIN_REPS,
    ):
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is not installed. Install with:\n"
                "  pip install httpx\n"
                "or install the API extra:\n"
                "  pip install 'newspaper-ocr[api]'"
            )

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        self.api_key_env = api_key_env
        self.prompt = prompt
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.extra_headers = dict(extra_headers or {})
        self.repetition_min_len = repetition_min_len
        self.repetition_min_reps = repetition_min_reps

        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _recognize_api(self, image: Image.Image) -> str:
        """POST one region crop and return the model's text."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = self._client.post(
            self.endpoint,
            headers=self._headers(),
            json={
                "model": self.model,
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
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ],
                "max_tokens": self.max_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def recognize(self, region: Region) -> Region:
        for attempt in range(self.max_retries + 1):
            try:
                text = self._recognize_api(region.image)
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


class OpenRouterRecognizer(OpenAiCompatRecognizer):
    """:class:`OpenAiCompatRecognizer` preset for OpenRouter.

    Points ``base_url`` at OpenRouter and reads the key from
    ``OPENROUTER_API_KEY``. A subclass (rather than a lambda) so the CLI's
    ``--model`` and :class:`~newspaper_ocr.Pipeline`'s ``recognizer_model`` still
    route to ``model`` via signature introspection.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        **kwargs,
    ):
        super().__init__(
            model=model, base_url=base_url, api_key_env=api_key_env, **kwargs
        )
