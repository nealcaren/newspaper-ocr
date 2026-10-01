"""CLI entry point for newspaper-ocr."""
from __future__ import annotations
import sys
from pathlib import Path
import click


@click.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--backend", "-b", default="tesseract",
              help="Recognition backend: tesseract, tesserocr, effocr, glm-ocr, "
                   "paddleocr-vl, mineru, mineru-vllm, mineru-http, openai, openrouter "
                   "(use --model to name the hosted model)")
@click.option("--detector", "-d", default="auto",
              help="Detection backend: auto (prefers doclayout_yolo, then paddlex, "
                   "then as_yolo), doclayout_yolo, paddlex, mineru, mineru-vllm, "
                   "mineru-http, as_yolo")
@click.option("--hole-fill-detector", default=None,
              help="Second detector whose boxes fill inked holes the main "
                   "detector missed (e.g. doclayout_yolo)")
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
@click.option("--fallback", default=None,
              help="Fallback recognizer for low-confidence lines (e.g. glm-ocr)")
@click.option("--fallback-threshold", default=70, type=float,
              help="Confidence threshold (0-100) below which fallback is used (default: 70)")
@click.option("--no-residual", is_flag=True,
              help="Disable the residual second pass (on by default for region recognizers)")
@click.option("--markup", default="plain", type=click.Choice(["plain", "raw"]),
              help="plain (default): strip HTML tables, LaTeX and Markdown from "
                   "VLM output; raw: keep the model's markup")
@click.option("--outdir", default=None,
              help="Output directory (default: stdout)")
@click.option("--rotate", default=0, type=click.Choice(["0", "90", "180", "270"]),
              help="PDF input: rotate each page clockwise before layout detection")
@click.option("--pages", "page_spec", default=None,
              help="PDF input: 1-based pages to read, e.g. 1-3,7 (default: all)")
@click.option("--dpi", default=300, type=int,
              help="PDF input: render resolution for pages with no embedded scan")
@click.option("--files-from", type=click.File("r"), default=None,
              help="Read input paths from this file, one per line ('-' for stdin)")
@click.option("--shard", default=None,
              help="Process only shard I of N inputs, as I/N (0-based), e.g. "
                   "$SLURM_ARRAY_TASK_ID/50")
@click.option("--skip-existing", is_flag=True,
              help="With --outdir: skip pages whose output file already exists")
@click.option("--log", "log_path", default=None, type=click.Path(),
              help="With --outdir: append one JSON line per page (status, seconds, "
                   "regions, chars, error) to this file")
@click.option("--input-root", default=None, type=click.Path(exists=True, file_okay=False),
              help="With --outdir: mirror input folders below this root under --outdir")
def main(images, backend, detector, hole_fill_detector, output, model, model_dir, mode, no_layout_processing, no_text_cleaning, spell_check, fallback, fallback_threshold, no_residual, markup, outdir, rotate, page_spec, dpi, files_from, shard, skip_existing, log_path, input_root):
    """OCR historical newspaper scans.

    Examples:
      newspaper-ocr page.jp2
      newspaper-ocr page.jp2 --backend tesserocr --output json
      newspaper-ocr *.jp2 --outdir results/ --output text
      newspaper-ocr page.jp2 --model news_gold_v2.traineddata
      newspaper-ocr issue.pdf --outdir results/   # one file per page
      newspaper-ocr --files-from pdfs.txt --shard 3/50 --outdir out/ \\
          --skip-existing --log out/log-3.jsonl     # one task of a job array

    PDF inputs (needs newspaper-ocr[pdf]) are read page by page; with --outdir
    each page is saved as <stem>_p001.txt, <stem>_p002.txt, ...

    With --outdir the run is batch-safe: each page is written atomically, a
    failing page is logged and skipped rather than ending the run, and the exit
    status is 1 if any page failed.
    """
    pages = _parse_pages(page_spec) if page_spec else None
    images = list(images)
    if files_from is not None:
        images += [ln.strip() for ln in files_from if ln.strip()]
    if not images:
        raise click.UsageError("Give input files or --files-from.")
    if shard is not None:
        from newspaper_ocr.batch import shard_inputs
        images = shard_inputs(images, _parse_shard(shard))
    if (skip_existing or log_path or input_root) and not outdir:
        raise click.UsageError("--skip-existing, --log and --input-root need --outdir.")

    from newspaper_ocr import Pipeline

    # Build recognizer with mode
    from newspaper_ocr.recognizers import RECOGNIZERS
    import inspect
    rec_cls = RECOGNIZERS.get(backend)
    rec_kwargs = {}
    params = inspect.signature(rec_cls.__init__).parameters
    if "mode" in params:
        rec_kwargs["mode"] = mode
    if model:
        # Route model arg to the right param
        if "model" in params:
            rec_kwargs["model"] = model
        elif "model_dir" in params:
            rec_kwargs["model_dir"] = model

    recognizer = rec_cls(**rec_kwargs)

    pipe = Pipeline(
        detector=detector,
        hole_fill_detector=hole_fill_detector,
        recognizer=recognizer,
        output=output,
        model_cache_dir=model_dir,
        layout_processing=not no_layout_processing,
        text_cleaning=not no_text_cleaning,
        spell_check=spell_check,
        fallback=fallback,
        fallback_threshold=fallback_threshold,
        residual_ocr=False if no_residual else "auto",
        markup=markup,
    )

    if outdir:
        from newspaper_ocr.batch import run_batch

        def progress(record):
            page = f" p{record['page']}" if record["page"] else ""
            line = f"{record['status']:7} {record['seconds']:7.1f}s {record['source']}{page}"
            if record.get("error"):
                line += f"  {record['error']}"
            click.echo(line, err=True)

        summary = run_batch(
            pipe, images, outdir, output, pages=pages, rotate=int(rotate), dpi=dpi,
            skip_existing=skip_existing, log_path=log_path, input_root=input_root,
            on_page=progress,
        )
        counts = ", ".join(f"{n} {k}" for k, n in sorted(summary.counts.items()))
        click.echo(f"Done in {summary.seconds:.0f}s: {counts or 'nothing to do'}", err=True)
        if summary.errors:
            sys.exit(1)
        return

    for image_path in images:
        if Path(image_path).suffix.lower() == ".pdf":
            from newspaper_ocr.pdf import page_images

            count = _pdf_page_count(image_path)
            numbers = range(count) if pages is None else pages
            if any(n >= count for n in numbers):
                raise click.BadParameter(
                    f"{Path(image_path).name} has only {count} pages",
                    param_hint="--pages")
            images_iter = page_images(image_path, dpi=dpi, rotate=int(rotate),
                                      pages=numbers)
            for number, image in zip(numbers, images_iter):
                click.echo(f"=== {Path(image_path).name} page {number + 1} ===",
                           err=True)
                click.echo(pipe.run(image))
        else:
            click.echo(pipe.ocr(image_path))


def _parse_shard(spec: str) -> tuple[int, int]:
    """Turn ``I/N`` into ``(I, N)``, checking 0 <= I < N."""
    try:
        index, count = (int(x) for x in spec.split("/"))
        if not 0 <= index < count:
            raise ValueError
    except ValueError:
        raise click.BadParameter(f"invalid shard {spec!r}; use I/N with 0 <= I < N",
                                 param_hint="--shard")
    return index, count


def _pdf_page_count(path: str) -> int:
    from newspaper_ocr.pdf import _require_pymupdf

    with _require_pymupdf().open(path) as doc:
        return doc.page_count


def _parse_pages(spec: str) -> list[int]:
    """Turn a 1-based spec like ``1-3,7`` into zero-based page numbers."""
    numbers: list[int] = []
    try:
        for part in spec.split(","):
            first, dash, last = part.strip().partition("-")
            start = int(first)
            end = int(last) if dash else start
            if start < 1 or end < start:
                raise ValueError
            numbers.extend(range(start - 1, end))
    except ValueError:
        raise click.BadParameter(f"invalid page spec {spec!r}; use e.g. 1-3,7",
                                 param_hint="--pages")
    return numbers
