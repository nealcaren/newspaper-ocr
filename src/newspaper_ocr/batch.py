"""Run a pipeline over many images and PDFs, safely enough to leave unattended.

A long OCR run gets killed: wall-time limits, preempted GPUs, a bad scan that
crashes a model.  :func:`run_batch` is built so that is cheap:

*Atomic, resumable output.*  Each page is written to a temporary file and
renamed into place, so a page file exists only if it is complete.  With
``skip_existing=True`` a rerun skips finished pages without even decoding them.

*One bad page doesn't sink the batch.*  Errors are caught per page, logged, and
counted; the run carries on.

*A per-page log.*  ``log_path`` gets one JSON line per page — status, seconds,
region count, characters, flagged regions, error — which is what you need to
find slow, empty or failing pages across a collection.

*Sharding.*  ``shard=(i, n)`` takes every n-th input starting at i, so each task
of a Slurm job array can be handed the same input list.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

EXTENSIONS = {"text": ".txt", "json": ".json", "hocr": ".hocr"}


@dataclass
class BatchSummary:
    """Counts of page outcomes: ``ok``, ``skipped`` and ``error``."""

    counts: Counter = field(default_factory=Counter)
    seconds: float = 0.0

    @property
    def errors(self) -> int:
        return self.counts["error"]


@dataclass
class _Page:
    source: Path
    page: int | None  # zero-based PDF page; None for a plain image
    output: Path


def shard_inputs(inputs: list, shard: tuple[int, int] | None) -> list:
    """Every n-th input starting at i, for ``shard=(i, n)``."""
    if shard is None:
        return list(inputs)
    index, count = shard
    if not 0 <= index < count:
        raise ValueError(f"shard index must be in [0, {count}); got {index}")
    return list(inputs)[index::count]


def output_path(source: Path, page: int | None, outdir: Path, fmt: str,
                input_root: Path | None = None) -> Path:
    """Where a page's output goes: ``<outdir>/[relative dirs/]<stem>[_pNNN].<ext>``.

    With *input_root*, the source's directories below it are mirrored under
    *outdir* (``1960/issue.pdf`` → ``outdir/1960/issue_p001.txt``), which keeps
    a large collection from landing in one flat directory.
    """
    stem = source.stem if page is None else f"{source.stem}_p{page + 1:03d}"
    folder = outdir
    if input_root is not None:
        folder = outdir / source.resolve().parent.relative_to(input_root.resolve())
    return folder / (stem + EXTENSIONS.get(fmt, ".txt"))


def run_batch(
    pipe,
    inputs: Iterable[str | Path],
    outdir: str | Path,
    fmt: str = "text",
    *,
    pages: list[int] | None = None,
    rotate: int = 0,
    dpi: int = 300,
    skip_existing: bool = False,
    log_path: str | Path | None = None,
    input_root: str | Path | None = None,
    on_page=None,
) -> BatchSummary:
    """OCR every image and PDF page in *inputs* into *outdir*.

    *pipe* is a :class:`~newspaper_ocr.pipeline.Pipeline` whose formatter
    matches *fmt*.  *on_page*, if given, is called with each log record — the
    CLI uses it for progress lines.
    """
    outdir = Path(outdir)
    root = Path(input_root) if input_root is not None else None
    summary = BatchSummary()
    start = time.monotonic()
    log = None
    if log_path:
        # The log commonly lives inside a not-yet-created outdir.
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8")
    try:
        for source in map(Path, inputs):
            for record in _run_source(pipe, source, outdir, fmt, pages, rotate, dpi,
                                      skip_existing, root):
                summary.counts[record["status"]] += 1
                if log:
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                if on_page:
                    on_page(record)
    finally:
        if log:
            log.close()
    summary.seconds = time.monotonic() - start
    return summary


def _run_source(pipe, source, outdir, fmt, pages, rotate, dpi, skip_existing, root
                ) -> Iterator[dict]:
    if source.suffix.lower() != ".pdf":
        yield _run_page(pipe, _Page(source, None, output_path(source, None, outdir, fmt, root)),
                        lambda: source, skip_existing)
        return

    from newspaper_ocr.pdf import _require_pymupdf, page_image

    try:
        doc = _require_pymupdf().open(str(source))
    except Exception as exc:
        yield _record(_Page(source, None, outdir), "error", 0.0, error=exc)
        return
    with doc:
        numbers = range(doc.page_count) if pages is None else pages
        for number in numbers:
            task = _Page(source, number, output_path(source, number, outdir, fmt, root))
            if number >= doc.page_count:
                yield _record(task, "error", 0.0, error=IndexError(
                    f"{source.name} has only {doc.page_count} pages"))
                continue
            yield _run_page(
                pipe, task,
                lambda n=number: page_image(doc, doc[n], dpi=dpi, rotate=rotate),
                skip_existing,
            )


def _run_page(pipe, task: _Page, load, skip_existing: bool) -> dict:
    if skip_existing and task.output.exists():
        return _record(task, "skipped", 0.0)
    t0 = time.monotonic()
    try:
        layout = pipe.analyze(load())
        result = pipe.formatter.format(layout)
        task.output.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a page file only ever exists complete, so an
        # interrupted run can't leave a truncated page that --skip-existing
        # would then treat as done.
        tmp = task.output.with_name(task.output.name + f".tmp{os.getpid()}")
        tmp.write_text(result, encoding="utf-8")
        os.replace(tmp, task.output)
    except Exception as exc:  # one bad page must not end the run
        return _record(task, "error", time.monotonic() - t0, error=exc)
    regions = layout.regions
    return _record(
        task, "ok", time.monotonic() - t0,
        regions=len(regions),
        chars=sum(len(r.text or "") for r in regions),
        flagged=sum(1 for r in regions if getattr(r, "status", "ok") != "ok"),
    )


def _record(task: _Page, status: str, seconds: float, error: BaseException | None = None,
            **stats) -> dict:
    record = {
        "source": str(task.source),
        "page": None if task.page is None else task.page + 1,
        "output": str(task.output),
        "status": status,
        "seconds": round(seconds, 2),
        **stats,
    }
    if error is not None:
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )[-2000:]
    return record
