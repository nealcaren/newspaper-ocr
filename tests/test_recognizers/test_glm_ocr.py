"""Tests for GLM-OCR recognizer."""

import pytest
from unittest.mock import patch

from newspaper_ocr.recognizers.glm_ocr import GlmOcrRecognizer
from newspaper_ocr.recognizers.base import RegionRecognizer


class TestGlmOcrImport:
    """Tests that work without GLM-OCR dependencies installed."""

    def test_class_exists_and_is_region_recognizer(self):
        assert issubclass(GlmOcrRecognizer, RegionRecognizer)

    def test_import_error_local_mode(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                GlmOcrRecognizer(mode="local")

    def test_import_error_api_mode(self):
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx"):
                GlmOcrRecognizer(mode="api")

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            GlmOcrRecognizer(mode="bogus")


class TestRepetitionDetection:
    """Test the static repetition helpers (no deps needed)."""

    def test_no_repetition(self):
        assert not GlmOcrRecognizer._has_repetition("Hello world this is normal text.")

    def test_detects_repetition(self):
        repeated = "The quick brown fox " * 10
        assert GlmOcrRecognizer._has_repetition(repeated)

    def test_truncate_repetition(self):
        repeated = "Hello world " * 5
        result = GlmOcrRecognizer._truncate_repetition(repeated)
        assert len(result) < len(repeated)
        assert "Hello world" in result

    def test_truncate_no_repetition(self):
        text = "Unique text here."
        assert GlmOcrRecognizer._truncate_repetition(text) == text

    def test_detects_repetition_not_anchored_at_start(self):
        """The loop usually starts partway in, after some good text."""
        text = "A genuine opening paragraph about the strike. " + "and so on and so on " * 8
        assert GlmOcrRecognizer._has_repetition(text)

    def test_prose_with_common_short_words_is_not_repetition(self):
        text = (
            "The strike committee met on Tuesday evening in the hall on Halsted "
            "Street, and the delegates from the west side locals reported that "
            "the men were holding firm against the new schedule of wages."
        )
        assert not GlmOcrRecognizer._has_repetition(text)

    def test_truncate_keeps_two_occurrences(self):
        """Production cuts after the second occurrence, not the first."""
        result = GlmOcrRecognizer._truncate_repetition("STRIKE AT THE MILL. " * 12)
        assert result == "STRIKE AT THE MILL. STRIKE AT THE MILL."

    def test_thresholds_are_configurable(self):
        from newspaper_ocr import repetition

        text = "repeat this phrase! " * 3
        assert not repetition.has_repetition(text)
        assert repetition.has_repetition(text, min_len=20, min_reps=3)


def _region():
    from PIL import Image
    from newspaper_ocr.models import BBox, Region

    return Region(
        bbox=BBox(0, 0, 100, 50),
        image=Image.new("RGB", (100, 50), "white"),
        label="text",
        lines=[],
    )


def _stub_recognizer(max_retries: int = 0) -> GlmOcrRecognizer:
    """A recognizer with no backend loaded, for exercising recognize() alone."""
    from newspaper_ocr import repetition

    rec = GlmOcrRecognizer.__new__(GlmOcrRecognizer)
    rec.mode = "api"
    rec.max_retries = max_retries
    rec.timeout = 25
    rec.repetition_min_len = repetition.MIN_LEN
    rec.repetition_min_reps = repetition.MIN_REPS
    rec._client = None
    return rec


class TestGlmOcrRecognize:
    """Test recognize() with mocked backends."""

    def test_recognize_sets_region_text(self):
        region = _region()

        recognizer = _stub_recognizer()
        recognizer._recognize_api = lambda img: "Recognized text"

        result = recognizer.recognize(region)
        assert result.text == "Recognized text"
        assert result.status == "ok"

    def test_recognize_handles_exception(self):
        region = _region()
        recognizer = _stub_recognizer()

        recognizer._recognize_api = lambda img: (_ for _ in ()).throw(
            RuntimeError("boom")
        )

        result = recognizer.recognize(region)
        assert result.text == ""
        assert result.status == "error"

    def test_timeout_sets_status_and_placeholder(self):
        from newspaper_ocr.errors import OcrTimeout
        from newspaper_ocr.recognizers.glm_ocr import TIMEOUT_TEXT

        region = _region()
        recognizer = _stub_recognizer()
        recognizer._recognize_api = lambda img: (_ for _ in ()).throw(
            OcrTimeout("too slow")
        )

        result = recognizer.recognize(region)
        assert result.text == TIMEOUT_TEXT
        assert result.status == "timeout"

    def test_third_party_timeout_is_classified_as_timeout(self):
        """httpx.TimeoutException & friends, recognised without importing httpx."""

        class ReadTimeout(Exception):
            pass

        region = _region()
        recognizer = _stub_recognizer()
        recognizer._recognize_api = lambda img: (_ for _ in ()).throw(
            ReadTimeout("read timed out")
        )

        assert recognizer.recognize(region).status == "timeout"

    def test_repetition_sets_status_and_truncates(self):
        region = _region()
        recognizer = _stub_recognizer()
        recognizer._recognize_api = lambda img: "STRIKE AT THE MILL. " * 12

        result = recognizer.recognize(region)
        assert result.status == "repetition"
        assert result.text == "STRIKE AT THE MILL. STRIKE AT THE MILL."

    def test_retries_before_giving_up(self):
        region = _region()
        recognizer = _stub_recognizer(max_retries=2)
        calls = []

        def _flaky(img):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("transient")
            return "Recovered text"

        recognizer._recognize_api = _flaky

        result = recognizer.recognize(region)
        assert result.text == "Recovered text"
        assert result.status == "ok"
        assert len(calls) == 3


@pytest.mark.skipif(
    not hasattr(__import__("signal"), "setitimer"),
    reason="SIGALRM-based guard is Unix-only",
)
class TestLocalTimeout:
    """The local-mode wall-clock guard (issue #1)."""

    def test_alarm_interrupts_a_hanging_call(self):
        import time

        from newspaper_ocr.errors import OcrTimeout
        from newspaper_ocr.recognizers.glm_ocr import _wall_clock_alarm

        start = time.monotonic()
        with pytest.raises(OcrTimeout):
            with _wall_clock_alarm(0.1):
                time.sleep(5)
        assert time.monotonic() - start < 2

    def test_alarm_is_disarmed_on_success(self):
        import signal
        import time

        from newspaper_ocr.recognizers.glm_ocr import _wall_clock_alarm

        previous = signal.getsignal(signal.SIGALRM)
        with _wall_clock_alarm(5) as armed:
            assert armed
        # Timer cleared and handler restored, so a later sleep is untouched.
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        assert signal.getsignal(signal.SIGALRM) is previous
        time.sleep(0.01)

    def test_alarm_is_a_noop_off_the_main_thread(self):
        import threading

        from newspaper_ocr.recognizers.glm_ocr import _wall_clock_alarm

        result = {}

        def _run():
            with _wall_clock_alarm(1) as armed:
                result["armed"] = armed

        t = threading.Thread(target=_run)
        t.start()
        t.join()
        assert result["armed"] is False


@pytest.fixture
def has_glm_ocr():
    try:
        from transformers import AutoProcessor  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")


class TestGlmOcrIntegration:
    """Tests that require transformers (skipped if not installed)."""

    def test_registry_registration(self, has_glm_ocr):
        from newspaper_ocr.recognizers import RECOGNIZERS

        assert "glm-ocr" in RECOGNIZERS._entries
