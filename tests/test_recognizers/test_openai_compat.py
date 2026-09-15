"""Tests for the OpenAI-compatible (bring-your-own-endpoint) recognizer."""

import pytest
from unittest.mock import patch

from PIL import Image

from newspaper_ocr.models import Region
from newspaper_ocr.recognizers.base import RegionRecognizer
from newspaper_ocr.recognizers.openai_compat import (
    OpenAiCompatRecognizer,
    OpenRouterRecognizer,
    TIMEOUT_TEXT,
)


def _region():
    return Region(bbox=None, image=Image.new("RGB", (40, 15), "white"), label="text")


class _FakeResp:
    def __init__(self, content, status_code=200, text="", usage=None):
        self._content = content
        self.status_code = status_code
        self.text = text
        self._usage = usage

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        body = {"choices": [{"message": {"content": self._content}}]}
        if self._usage is not None:
            body["usage"] = self._usage
        return body


class _FakeClient:
    """Captures the request and returns a canned response (or raises)."""

    def __init__(self, content="transcribed text", raises=None):
        self.content = content
        self.raises = raises
        self.calls = []

    def post(self, url, headers, json):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.raises is not None:
            raise self.raises
        return _FakeResp(self.content)


class TestConstruction:
    def test_is_region_recognizer(self):
        assert issubclass(OpenAiCompatRecognizer, RegionRecognizer)
        assert issubclass(OpenRouterRecognizer, OpenAiCompatRecognizer)

    def test_import_error_without_httpx(self):
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx"):
                OpenAiCompatRecognizer(model="gpt-4o-mini", api_key="sk-test")

    def test_endpoint_appends_chat_completions(self):
        r = OpenAiCompatRecognizer(model="m", api_key="sk-test")
        assert r.endpoint == "https://api.openai.com/v1/chat/completions"

    def test_base_url_trailing_slash_stripped(self):
        r = OpenAiCompatRecognizer(
            model="m", api_key="sk-test", base_url="http://localhost:1234/v1/"
        )
        assert r.endpoint == "http://localhost:1234/v1/chat/completions"

    def test_api_key_read_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "sk-from-env")
        r = OpenAiCompatRecognizer(model="m", api_key_env="MY_KEY")
        assert r.api_key == "sk-from-env"
        assert r._headers()["Authorization"] == "Bearer sk-from-env"

    def test_openrouter_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        r = OpenRouterRecognizer()
        assert r.endpoint == "https://openrouter.ai/api/v1/chat/completions"
        assert r.model == "openai/gpt-4o-mini"
        assert r.api_key == "sk-or"


class TestRecognize:
    def test_returns_text_and_ok_status(self):
        r = OpenAiCompatRecognizer(model="gpt-4o-mini", api_key="sk-test")
        r._client = _FakeClient(content="  hello  ")
        reg = r.recognize(_region())
        assert reg.text == "hello"
        assert reg.status == "ok"

    def test_request_payload_shape(self):
        r = OpenAiCompatRecognizer(
            model="gpt-4o-mini", api_key="sk-test", extra_headers={"X-Title": "n"}
        )
        client = _FakeClient()
        r._client = client
        r.recognize(_region())
        call = client.calls[0]
        assert call["url"].endswith("/chat/completions")
        assert call["json"]["model"] == "gpt-4o-mini"
        assert call["headers"]["Authorization"] == "Bearer sk-test"
        assert call["headers"]["X-Title"] == "n"
        content = call["json"]["messages"][0]["content"]
        assert [p["type"] for p in content] == ["image_url", "text"]
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_null_content_yields_empty_text(self):
        # Some providers/models return content: null with a 200 (empty
        # completion or a filtered response); it must not crash on .strip().
        r = OpenAiCompatRecognizer(model="m", api_key="sk-test")
        r._client = _FakeClient(content=None)
        reg = r.recognize(_region())
        assert reg.text == ""
        assert reg.status == "ok"

    def test_error_yields_error_status(self):
        r = OpenAiCompatRecognizer(model="m", api_key="sk-test", max_retries=0)
        r._client = _FakeClient(raises=RuntimeError("boom"))
        reg = r.recognize(_region())
        assert reg.text == ""
        assert reg.status == "error"

    def test_timeout_yields_timeout_status(self):
        r = OpenAiCompatRecognizer(model="m", api_key="sk-test", max_retries=0)
        r._client = _FakeClient(raises=TimeoutError("slow"))
        reg = r.recognize(_region())
        assert reg.text == TIMEOUT_TEXT
        assert reg.status == "timeout"

    def test_token_param_defaults_to_max_tokens(self):
        r = OpenAiCompatRecognizer(model="m", api_key="sk-test")
        client = _FakeClient()
        r._client = client
        r.recognize(_region())
        assert "max_tokens" in client.calls[0]["json"]
        assert r._token_param == "max_tokens"

    def test_token_param_flips_on_400(self):
        # First POST 400s asking for max_completion_tokens; recognizer flips and
        # retries, then locks the new field name for subsequent calls.
        r = OpenAiCompatRecognizer(model="m", api_key="sk-test")

        class FlipClient:
            def __init__(self):
                self.calls = []

            def post(self, url, headers, json):
                self.calls.append(json)
                if "max_tokens" in json:
                    return _FakeResp(
                        None,
                        status_code=400,
                        text="Unsupported parameter: use 'max_completion_tokens'",
                    )
                return _FakeResp("ok text")

        client = FlipClient()
        r._client = client
        reg = r.recognize(_region())
        assert reg.text == "ok text"
        assert reg.status == "ok"
        assert r._token_param == "max_completion_tokens"
        # first call max_tokens (400), retry max_completion_tokens (200)
        assert "max_tokens" in client.calls[0]
        assert "max_completion_tokens" in client.calls[1]
        # a second region reuses the locked field — no repeat probe
        r.recognize(_region())
        assert "max_completion_tokens" in client.calls[2]
        assert len(client.calls) == 3

    def test_explicit_token_param_is_not_probed(self):
        r = OpenAiCompatRecognizer(
            model="m", api_key="sk-test", token_param="max_completion_tokens"
        )
        client = _FakeClient()
        r._client = client
        r.recognize(_region())
        assert "max_completion_tokens" in client.calls[0]["json"]

    def test_usage_is_tracked(self):
        r = OpenAiCompatRecognizer(model="m", api_key="sk-test")
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 40},
        }

        class UsageClient:
            def __init__(self):
                self.calls = []

            def post(self, url, headers, json):
                self.calls.append(json)
                return _FakeResp("hi", usage=usage)

        r._client = UsageClient()
        r.recognize(_region())
        r.recognize(_region())
        assert r.last_usage == usage
        assert r.usage_totals["requests"] == 2
        assert r.usage_totals["prompt_tokens"] == 200
        assert r.usage_totals["completion_tokens"] == 40
        assert r.usage_totals["cached_prompt_tokens"] == 80

    def test_cost_uses_prices_and_cached_rate(self):
        r = OpenAiCompatRecognizer(model="m", api_key="sk-test")
        r.usage_totals = {
            "requests": 1,
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "total_tokens": 2_000_000,
            "cached_prompt_tokens": 0,
        }
        # 1M input @ $0.20 + 1M output @ $1.20 = $1.40
        assert r.cost(0.20, 1.20) == pytest.approx(1.40)

        # with caching: 200k of the input tokens cached at $0.02
        r.usage_totals["prompt_tokens"] = 1_000_000
        r.usage_totals["cached_prompt_tokens"] = 200_000
        # 800k @ 0.20 + 200k @ 0.02 + 1M @ 1.20 = 0.16 + 0.004 + 1.20
        assert r.cost(0.20, 1.20, cached_input_per_mtok=0.02) == pytest.approx(1.364)

    def test_retries_then_succeeds(self):
        r = OpenAiCompatRecognizer(model="m", api_key="sk-test", max_retries=2)

        class Flaky:
            def __init__(self):
                self.n = 0

            def post(self, url, headers, json):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("transient")
                return _FakeResp("recovered")

        r._client = Flaky()
        reg = r.recognize(_region())
        assert reg.text == "recovered"
        assert reg.status == "ok"
