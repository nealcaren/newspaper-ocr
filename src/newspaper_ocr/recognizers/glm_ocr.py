from __future__ import annotations

import re

from PIL import Image

from newspaper_ocr.models import Region
from newspaper_ocr.recognizers.base import RegionRecognizer


class GlmOcrRecognizer(RegionRecognizer):
    """GLM-OCR vision-language model. Requires GPU or MLX server.

    Two modes:
    - mode="api": Connect to running MLX/vLLM server (default)
    - mode="local": Load model directly via transformers (GPU required)
    """

    def __init__(
        self,
        mode: str = "api",
        api_url: str = "http://localhost:8080/v1/chat/completions",
        model_id: str = "zai-org/GLM-OCR",
        mlx_model_id: str = "mlx-community/GLM-OCR-bf16",
        timeout: int = 25,
        max_retries: int = 2,
    ):
        self.mode = mode
        self.api_url = api_url
        self.model_id = model_id
        self.mlx_model_id = mlx_model_id
        self.timeout = timeout
        self.max_retries = max_retries

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
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto"
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

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=4096)

        return self._processor.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _has_repetition(text: str, min_len: int = 10, threshold: int = 3) -> bool:
        """Detect pathological repetition in OCR output."""
        if len(text) < min_len * threshold:
            return False
        # Check for repeated substrings of various lengths
        for length in range(min_len, len(text) // threshold + 1):
            pattern = re.escape(text[:length])
            matches = len(re.findall(pattern, text))
            if matches >= threshold:
                return True
        return False

    @staticmethod
    def _truncate_repetition(text: str, min_len: int = 10) -> str:
        """Truncate text at the first detected repetition."""
        for length in range(min_len, len(text) // 2 + 1):
            candidate = text[:length]
            rest = text[length:]
            if rest.startswith(candidate):
                return candidate.strip()
        return text.strip()

    def recognize(self, region: Region) -> Region:
        for attempt in range(self.max_retries + 1):
            try:
                if self.mode == "api":
                    text = self._recognize_api(region.image)
                else:
                    text = self._recognize_local(region.image)

                if not self._has_repetition(text):
                    region.text = text
                    return region
                if attempt < self.max_retries:
                    continue
                region.text = self._truncate_repetition(text)
                return region
            except Exception:
                if attempt == self.max_retries:
                    region.text = ""
                    return region
        return region
