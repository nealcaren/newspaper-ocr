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
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


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
