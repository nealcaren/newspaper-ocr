from __future__ import annotations
from pathlib import Path
from newspaper_ocr.models import Line
from newspaper_ocr.recognizers.base import LineRecognizer

# Default HuggingFace model location
_HF_REPO = "NealCaren/kraken-newspaper-ocr"
_HF_FILENAME = "kraken_pre1930_combined.mlmodel"


class KrakenRecognizer(LineRecognizer):
    def __init__(
        self,
        model: str | Path | None = None,
        model_cache_dir: str | Path | None = None,
    ):
        """
        Kraken OCR recognizer for historical newspaper text.

        Parameters
        ----------
        model : str or Path, optional
            Path to a local .mlmodel file.  When *None*, downloads
            the pre-trained model from HuggingFace on first use.
        model_cache_dir : str or Path, optional
            Directory for caching downloaded models.
        """
        try:
            from kraken.lib.models import load_any
        except ImportError:
            raise ImportError(
                "kraken is not installed. Install with:\n"
                "  pip install 'newspaper-ocr[kraken]'"
            )

        if model is None:
            model = self._download_model(model_cache_dir)

        self._model = load_any(str(model))

    @staticmethod
    def _download_model(cache_dir: str | Path | None = None) -> Path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_HF_FILENAME,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return Path(path)

    def recognize(self, line: Line) -> Line:
        from kraken.rpred import rpred
        from kraken.containers import Segmentation, BBoxLine

        img = line.image
        w, h = img.size
        seg = Segmentation(
            type="bbox",
            imagename="line.png",
            lines=[BBoxLine(id="l1", bbox=(0, 0, w, h))],
            text_direction="horizontal-lr",
            script_detection=False,
        )
        text = ""
        for record in rpred(self._model, img, seg):
            text = record.prediction

        line.text = text
        line.confidence = 0.9 if text else 0.0
        return line
