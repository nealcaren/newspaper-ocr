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
    token_param:
        Name of the token-limit request field. Left as ``None`` (the default) it
        auto-detects: it sends ``max_tokens`` and, if the endpoint rejects that
        with a 400 asking for ``max_completion_tokens`` (o1 / gpt-5 family),
        flips once and remembers the choice. Pass an explicit name to pin it and
        skip the probe.
    extra_headers:
        Additional HTTP headers merged into every request — handy for
        OpenRouter's optional ``HTTP-Referer`` / ``X-Title`` attribution.
    repetition_min_len / repetition_min_reps:
        Loop-detector thresholds, shared with the other VLM backends.

    Attributes
    ----------
    last_usage:
        The ``usage`` block from the most recent response, or ``None``.
    usage_totals:
        Running token/request totals across this instance. Pair with
        :meth:`cost` and the model's published per-1M-token prices to estimate
        spend — e.g. ``rec.cost(0.20, 1.20)`` for a model at $0.20/$1.20.
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
        token_param: str | None = None,
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
        # Name of the token-limit field. Newer OpenAI models (o1, gpt-5 family)
        # reject "max_tokens" and require "max_completion_tokens"; older models
        # and many third-party endpoints only accept "max_tokens". Default to
        # "max_tokens" and auto-switch on the first 400 that asks for the other.
        self._token_param = token_param or "max_tokens"
        self._token_param_locked = token_param is not None
        self.extra_headers = dict(extra_headers or {})
        self.repetition_min_len = repetition_min_len
        self.repetition_min_reps = repetition_min_reps

        #: The ``usage`` block from the most recent response (or ``None``).
        self.last_usage: dict | None = None
        #: Running totals across every billed response this instance made.
        #: ``requests`` counts responses that reported usage; the token fields
        #: sum the corresponding ``usage`` values.
        self.usage_totals: dict[str, int] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
        }

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

        def _post():
            return self._client.post(
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
                    self._token_param: self.max_tokens,
                },
            )

        resp = _post()
        # Newer OpenAI models reject "max_tokens" and want "max_completion_tokens"
        # (and vice-versa for some endpoints). On that specific 400, flip the
        # field name once, remember it, and retry — so the switch costs one round
        # trip on the first call and nothing thereafter.
        if (
            resp.status_code == 400
            and not self._token_param_locked
            and "max_completion_tokens" in resp.text
        ):
            self._token_param = (
                "max_completion_tokens"
                if self._token_param == "max_tokens"
                else "max_tokens"
            )
            self._token_param_locked = True
            resp = _post()

        resp.raise_for_status()
        data = resp.json()
        self._record_usage(data.get("usage"))
        # Some providers/models return content: null (empty completion) with a
        # 200; treat that as empty text rather than crashing on .strip().
        content = data["choices"][0]["message"].get("content")
        return (content or "").strip()

    def _record_usage(self, usage: dict | None) -> None:
        """Accumulate the ``usage`` block from a response, if present.

        Providers vary in what they report; missing fields are treated as zero
        and unknown extras are ignored. ``cached_prompt_tokens`` is read from the
        nested ``prompt_tokens_details.cached_tokens`` that OpenAI-style
        responses use.
        """
        if not usage:
            return
        self.last_usage = usage
        self.usage_totals["requests"] += 1
        self.usage_totals["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        self.usage_totals["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        self.usage_totals["total_tokens"] += usage.get("total_tokens", 0) or 0
        details = usage.get("prompt_tokens_details") or {}
        self.usage_totals["cached_prompt_tokens"] += details.get("cached_tokens", 0) or 0

    def cost(self, input_per_mtok: float, output_per_mtok: float,
             cached_input_per_mtok: float | None = None) -> float:
        """Estimate spend so far from :attr:`usage_totals` and per-1M-token prices.

        Cached prompt tokens are billed at ``cached_input_per_mtok`` when given
        (they are otherwise counted at the full input rate). Prices are the
        published per-1M-token figures for the model.
        """
        totals = self.usage_totals
        cached = totals["cached_prompt_tokens"]
        uncached_in = totals["prompt_tokens"] - cached
        rate_in = uncached_in / 1_000_000 * input_per_mtok
        rate_cached = (
            cached / 1_000_000
            * (cached_input_per_mtok if cached_input_per_mtok is not None else input_per_mtok)
        )
        rate_out = totals["completion_tokens"] / 1_000_000 * output_per_mtok
        return rate_in + rate_cached + rate_out

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
