"""CLI entry point for newspaper-ocr."""
from __future__ import annotations
import sys
from pathlib import Path
import click


@click.command()
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--backend", "-b", default="tesseract",
              help="Recognition backend: tesseract, tesserocr, effocr")
@click.option("--detector", "-d", default="as_yolo",
              help="Detection backend: as_yolo, paddlex")
@click.option("--output", "-o", default="text",
              help="Output format: text, json, hocr")
@click.option("--model", "-m", default=None,
              help="Custom model path (e.g., traineddata for Tesseract)")
@click.option("--model-dir", default=None,
              help="Model cache directory")
@click.option("--mode", default="region",
              help="Recognition mode: line or region (default: region)")
@click.option("--no-layout-processing", is_flag=True,
              help="Disable reading order post-processing")
@click.option("--no-text-cleaning", is_flag=True,
              help="Disable dehyphenation and line-joining post-processing")
@click.option("--spell-check", is_flag=True, default=False,
              help="Enable SymSpell spell correction post-processing (off by default)")
@click.option("--outdir", default=None,
              help="Output directory (default: stdout)")
def main(images, backend, detector, output, model, model_dir, mode, no_layout_processing, no_text_cleaning, spell_check, outdir):
    """OCR historical newspaper scans.

    Examples:
      newspaper-ocr page.jp2
      newspaper-ocr page.jp2 --backend tesserocr --output json
      newspaper-ocr *.jp2 --outdir results/ --output text
      newspaper-ocr page.jp2 --model news_gold_v2.traineddata
    """
    from newspaper_ocr import Pipeline

    # Build recognizer with mode
    from newspaper_ocr.recognizers import RECOGNIZERS
    rec_cls = RECOGNIZERS.get(backend)
    rec_kwargs = {"mode": mode}
    if model:
        # Route model arg to the right param
        import inspect
        params = inspect.signature(rec_cls.__init__).parameters
        if "model" in params:
            rec_kwargs["model"] = model
        elif "model_dir" in params:
            rec_kwargs["model_dir"] = model

    recognizer = rec_cls(**rec_kwargs)

    pipe = Pipeline(
        detector=detector,
        recognizer=recognizer,
        output=output,
        model_cache_dir=model_dir,
        layout_processing=not no_layout_processing,
        text_cleaning=not no_text_cleaning,
        spell_check=spell_check,
    )

    for image_path in images:
        result = pipe.ocr(image_path)

        if outdir:
            out_path = Path(outdir) / (Path(image_path).stem + _ext(output))
            Path(outdir).mkdir(parents=True, exist_ok=True)
            out_path.write_text(result)
            click.echo(f"Saved: {out_path}", err=True)
        else:
            click.echo(result)


def _ext(fmt: str) -> str:
    return {"text": ".txt", "json": ".json", "hocr": ".hocr"}.get(fmt, ".txt")
