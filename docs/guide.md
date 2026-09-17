# newspaper-ocr — detailed guide

Full reference for the three-phase pipeline, every backend, and the output
formats. The [README](../README.md) covers install, quick start, and which
detector/recognizer to pick; this page has the rest.

## PDF Input

Most scans arrive as multi-page PDFs rather than loose images.

```python
pipe = Pipeline(recognizer="glm-ocr", output="json")

# One formatted result per page
pages = pipe.ocr_pdf("issue.pdf")

# Sideways broadsheet, turned clockwise before layout detection
pages = pipe.ocr_pdf("industrial-worker-1912.pdf", rotate=90)
```

Requires: `pip install "newspaper-ocr[pdf]"`

Each page is taken from its **largest embedded image**, not the first one. Pages
scanned by Google carry a small "Digitized by Google" strip ahead of the page
image, so taking `images[0]` yields a few-hundred-pixel sliver instead of the
broadsheet. Pages with no embedded image at all (born-digital, vector text) are
rendered at `dpi` instead, 300 by default.

`rotate` accepts 0, 90, 180 or 270 **clockwise**, applied before layout
detection — a sideways page produces nothing usable, and titles in the same
collection are sometimes scanned in opposite directions, so it is a per-run
choice rather than a constant.

To stream pages without holding every page's output in memory, use the
front-end directly:

```python
from newspaper_ocr.pdf import page_images

for i, image in enumerate(page_images("issue.pdf", rotate=90)):
    Path(f"page-{i:03d}.json").write_text(pipe.run(image))
```

`page_images` also takes `pages=` to select a subset (zero-based) and
`prefer_embedded=False` to always render.

## Phase 1: Layout

Three detection backends, plus battle-tested newspaper layout post-processing.

### Detectors

| Detector | What it finds | Speed | Best for |
|----------|--------------|-------|----------|
| `doclayout_yolo` **(recommended)** | Regions only (10 categories) | varies | Newspapers/broadsheets; region-level OCR — best accuracy (DocLayout-YOLO, 1280px checkpoint) |
| `paddlex` | Regions only (20 categories) | varies | Newspapers/broadsheets; region-level OCR, detailed layout analysis |
| `as_yolo` | Regions + lines | ~8s/page | Line-level OCR (Tesseract, EffOCR) on simple layouts |

The default is **`detector="auto"`**, which prefers `doclayout_yolo` when it is
installed, then `paddlex`, and otherwise falls back to `as_yolo` (with a warning).
The **detector — not the recognizer — is the dominant factor** on dense newspaper
pages (see the results table in the [README](../README.md)). DocLayout-YOLO
proposes finer, more complete regions, so it needs no residual recovery — but it
makes more recognizer calls (≈2× slower). Install the recommended detector with
`pip install "newspaper-ocr[doclayout]"` (or `[paddlex]`).

The DocLayout-YOLO checkpoints are the official ones from the Hugging Face Hub —
[`juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501`](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501)
(default) and `juliozhao/DocLayout-YOLO-DocStructBench` (1024px) — downloaded and
cached on first use.

### Layout Processing

Ported from the [Dangerous Press](https://dangerouspress.org) production pipeline. Applied automatically after detection:

1. **Filter** low-confidence detections
2. **Rescue** missed regions in gaps between accepted detections
3. **Deduplicate** overlapping regions (three-pass: contained, title-text, near-duplicate)
4. **Fill column gaps** using geometric column detection
5. **Reading order** — column-aware sorting (full-width headers first, then column-by-column)
6. **Merge** vertically adjacent blocks into coherent regions
7. **Drop empty text regions** — text-labeled regions the line detector found nothing in

Disable with `layout_processing=False`.

Stage 7 runs only when a line detector actually ran, which the detector reports
as `PageLayout.lines_detected`. A text region with no lines is a layout false
positive when something looked and found nothing; when nothing looked — a
region-only detector such as `paddlex`, or `skip_lines=True` — every region is
line-less, so the stage is skipped and those regions go on to region-level OCR
instead. The flag defaults to `False`: a detector has to opt in, because letting
a false positive through costs one wasted OCR call while wrongly dropping a
region loses real text.

The tuned constants (column `gap_thresh`, the narrow-column merge, the merge
height cap, the confidence bands) are a 1:1 port of a specific revision of the
production pipeline, pinned as `newspaper_ocr.PIPELINE_REFERENCE_TAG`
(`2025-03-07-col-fix`). Diff against that tag before changing them — drift here
changes column segmentation, and therefore the output text, for every page.

## Phase 2: OCR

Recognition backends with different speed/accuracy tradeoffs.

| Backend | Mode | Speed | CER* | How it works |
|---------|------|-------|------|-------------|
| `tesseract` (eng) | line | ~106s | 9.0% | Stock Tesseract, LSTM sequence model |
| `tesseract` (news_combo_fast) | line | ~93s | 2.9% | **Bundled fine-tuned model** |
| `tesseract` | region | ~38s | — | Subprocess per region, Tesseract's own line segmentation |
| `tesserocr` | line | ~26s | — | C API bindings, no subprocess overhead |
| `tesserocr` | region | ~25s | — | C API, region-level |
| `kraken` | line | ~10s | 3.5% | Kraken LSTM, ~10x faster than Tesseract |
| `trocr` | line | ~35s | 3.6% | Fine-tuned TrOCR, GPU recommended |
| `glm-ocr` | region | ~300s | 1.7% | GLM-OCR VLM, GPU recommended |
| `lightonocr` | region | ~500s | **1.1%** | LightOnOCR-2-1B VLM, GPU required |
| `paddleocr-vl` | region | varies | — | PaddleOCR-VL VLM, GPU recommended; good fallback for regions another model failed on |
| `effocr` | line | ~50s | 11.2% | Contrastive char/word matching, ONNX |
| `openai` / `openrouter` | region | varies | — | Any OpenAI-compatible endpoint — bring your own hosted model |

*CER measured on pre-1930 newspaper text at R2 (35%) resolution. Times on a single newspaper page (~1,100 lines).

### Bundled Fine-Tuned Model

The package ships with `news_combo_fast` — a quantized Tesseract model (1.4MB) fine-tuned on ~60K lines of pre-1930 historical newspaper text. It achieves **2.9% CER** vs 9.0% for the stock `eng` model, with no speed penalty.

```python
# Use the fine-tuned model (recommended for historical newspapers)
pipe = Pipeline(recognizer="tesseract", recognizer_model="news_combo_fast")
```

### Kraken Backend

Kraken is ~10x faster than Tesseract per line with comparable accuracy (3.5% CER). The pre-trained model downloads automatically from HuggingFace on first use (~3MB).

```python
pipe = Pipeline(recognizer="kraken")
```

Requires: `pip install "newspaper-ocr[kraken]"`

### LightOnOCR Backend

LightOnOCR-2-1B achieves the lowest CER (1.1%) on historical newspaper text. Requires GPU (CUDA or MPS).

```python
pipe = Pipeline(recognizer="lightonocr")
```

Requires: `pip install "newspaper-ocr[lightonocr]"`

### GLM-OCR Backend

```python
pipe = Pipeline(recognizer="glm-ocr")
```

Runs either against a local MLX/vLLM server (`mode="api"`, the default) or
directly through transformers on a GPU (`mode="local"`).

```python
from newspaper_ocr.recognizers.glm_ocr import GlmOcrRecognizer

pipe = Pipeline(
    recognizer=GlmOcrRecognizer(
        mode="local",
        timeout=120,             # per-region wall-clock budget, both modes
        max_retries=2,
        repetition_min_len=20,   # loop detector, production defaults
        repetition_min_reps=5,
    )
)
```

`timeout` is enforced in local mode as well as API mode, so a pathological
region can't hang a whole batch: generation is guarded by `SIGALRM` where it is
available, with a between-token deadline as a portable backstop. A region that
exhausts its retries gets the text `[OCR timeout]` and `status="timeout"` rather
than silently empty text. In local mode on a CUDA GPU the model loads with
`.to("cuda")` (needs no `accelerate`); the default 120s budget suits large
newspaper column crops. (Both were fixed in 0.8.1 — earlier versions returned
empty output on GPU.)

### Bring your own OCR device

The built-in backends run models locally, but hosted models change fast and you
may want to point the pipeline at your own. There are two ways in — no fork
required.

**1. Any OpenAI-compatible endpoint (`openai` / `openrouter`).** These region
recognizers speak the OpenAI vision chat-completions protocol, so the same
backend works against OpenAI, OpenRouter, Together, Groq, or a local
vLLM/LM-Studio/Ollama server — you just name a model and point at a `base_url`.
Nothing is downloaded; the model lives behind the endpoint.

```bash
# CLI: --model names the hosted model; the key comes from the env var
export OPENAI_API_KEY=sk-...
newspaper-ocr page.jp2 --backend openai --model gpt-4o-mini --output text

export OPENROUTER_API_KEY=sk-or-...
newspaper-ocr page.jp2 --backend openrouter --model "openai/gpt-4o-mini"
```

```python
from newspaper_ocr import Pipeline
from newspaper_ocr.recognizers.openai_compat import OpenAiCompatRecognizer

pipe = Pipeline(
    recognizer=OpenAiCompatRecognizer(
        base_url="https://openrouter.ai/api/v1",   # or your own server
        model="openai/gpt-4o-mini",
        api_key_env="OPENROUTER_API_KEY",          # key read from the environment
        prompt="Transcribe all text exactly as it appears.",  # tune per language/document
        timeout=60,
        max_retries=2,
        extra_headers={"X-Title": "newspaper-ocr"},  # e.g. OpenRouter attribution
    ),
    output="text",
)
```

Requires: `pip install "newspaper-ocr[api]"`. It inherits the same retry,
timeout, and repetition-loop handling as the other region recognizers. The
token-limit field is auto-detected — it sends `max_tokens` and transparently
switches to `max_completion_tokens` for models that require it (o1 / gpt-5
family).

Token usage is tracked per instance so you can price a run. `rec.last_usage`
holds the most recent response's `usage`, `rec.usage_totals` accumulates across
the run, and `rec.cost(input_per_mtok, output_per_mtok)` estimates spend:

```python
rec = OpenAiCompatRecognizer(model="gpt-5.6-luna")
Pipeline(recognizer=rec).analyze(page)   # RGB image
print(rec.usage_totals)                  # {'requests': 23, 'prompt_tokens': 51502, ...}
print(rec.cost(0.20, 1.20))              # ~$0.02 for a full page at that model's price
```

Pass `extra_body=` to add fields to every request (e.g. `{"temperature": 0}`,
or OpenRouter's `{"usage": {"include": True}}` to get billed cost back). When the
endpoint returns a `cost` (OpenRouter does), it accumulates in
`rec.usage_totals["reported_cost"]` — exact dollars, no price table needed:

```python
rec = OpenRouterRecognizer(
    model="google/gemini-3.5-flash-lite",
    extra_body={"usage": {"include": True}},
)
Pipeline(detector="paddlex", recognizer=rec).analyze(page)
print(rec.usage_totals["reported_cost"])   # e.g. 0.0318 — actual OpenRouter spend
```

**2. Plug in any function.** For anything not OpenAI-shaped, pass a callable that
takes a `PIL.Image` and returns text. The pipeline wraps it as a region
recognizer (with the same error/timeout/loop handling), so it needs no
subclassing:

```python
from newspaper_ocr import Pipeline

def my_ocr(image):
    # call your own service, model, or API however you like
    return call_my_service(image)

pipe = Pipeline(recognizer=my_ocr, output="text")
```

For more control (subclass a recognizer, or register a named backend), see
`RegionRecognizer`/`LineRecognizer` in `newspaper_ocr.recognizers.base` and the
`RECOGNIZERS` registry in `newspaper_ocr.recognizers`.

### Recovering failed regions (splitting + fallback)

For region-level recognizers, two options add a recovery ladder for regions the
primary model fails on:

```python
pipe = Pipeline(
    recognizer="glm-ocr",         # primary
    fallback="paddleocr-vl",      # backup model for regions the primary failed on
    chunk_tall_regions=True,      # split tall regions that time out and re-OCR the bands
)
```

The ladder runs **primary → chunked re-OCR (same model) → fallback (different
model)**:

- **`chunk_tall_regions`** — when the primary times out on a region taller than
  `chunk_height` (default 500px), the crop is split into overlapping vertical
  bands (`chunk_overlap`, default 50px), each band is re-OCR'd with the *same*
  recognizer, and the texts are stitched back together (overlap de-duplicated).
  The region becomes `ok` if every band read, or `chunked_partial` if some band
  still timed out.
- **`fallback`** — any region left in a failure state is re-OCR'd by the fallback
  recognizer, **do-no-harm**: for `timeout`/`error` regions any usable read is
  taken; for partial regions (`repetition`/`chunked_partial`) the fallback text
  replaces the primary's only if it is a clean `ok` read. The original text is
  preserved in `region.text_primary` and the fallback engine recorded in
  `region.engine`.

`paddleocr-vl` is the intended fallback — a different VLM often succeeds where the
primary looped or timed out. It runs on a single CUDA GPU (as of 0.8.1); it is
slower than GLM-OCR but slightly more accurate.

One caveat on where you run it: `SIGALRM` can only be armed on the main thread
of a Unix process, and that is what interrupts a hung call mid-forward-pass. In
a worker thread (or on Windows) only the between-token deadline applies —
generation still stops at the budget, but a call that hangs *inside* a single
forward pass is reported as a timeout only once it returns. Run batches on the
main thread if you need hangs bounded rather than just detected.

The loop detector slides windows across the whole region text and counts
occurrences; on a hit the text is cut just after the second occurrence of the
repeated phrase and the region is marked `status="repetition"`. Defaults match
the production pipeline (`2025-03-07-col-fix`); lower `repetition_min_reps` to
catch loops sooner, raise `repetition_min_len` if legitimately repeated short
phrases are being clipped.

Requires: `pip install "newspaper-ocr[glm-ocr]"`

See [dangerouspress-ocr-finetune](https://github.com/nealcaren/ocr-finetune) for the training pipeline.

### Residual second pass (recover detector misses)

The recovery ladder above fixes regions the recognizer *failed on*. `ResidualOcr`
fixes a different error: text the **detector never boxed at all** — whole
mastheads, side columns, or inter-block strips the recognizer therefore never
sees. Once a strong recognizer saturates precision, this undetected text is the
dominant remaining error (recall).

It runs after recognition and is **non-destructive** (returns a new
`PageLayout`). As of 0.7.0 it is **on by default for region-level recognizers**
(GLM-OCR and other VLMs), where it is validated and do-no-harm gated; it stays off
for line recognizers such as Tesseract. Control it with `residual_ocr`:

```python
from newspaper_ocr import Pipeline

Pipeline(recognizer="glm-ocr")                       # residual ON (auto: region recognizer)
Pipeline(recognizer="glm-ocr", residual_ocr=False)   # opt out
Pipeline(recognizer="tesseract")                     # residual OFF (auto: line recognizer)
Pipeline(recognizer="tesseract", residual_ocr=True)  # force on (uses recognize_region)
```

`residual_ocr` accepts `"auto"` (default — on for region recognizers), `True`
(force on for any region-capable recognizer), or `False`. On the CLI, region
recognizers get it automatically; pass `--no-residual` to opt out.

To run it by hand (e.g. after a custom `analyze`), it's also a standalone pass:

```python
from newspaper_ocr.residual_ocr import ResidualOcr

pipe = Pipeline(recognizer="glm-ocr", residual_ocr=False)  # disable the built-in
residual = ResidualOcr(recognizer=pipe.recognizer)
layout = residual.recover(pipe.analyze("page.jpg"))
```

How it works: mask every pass-1 region box, find the leftover ink, cut it into
column-shaped blocks (a recursive XY-cut on the ink projections, with a valley
split for multi-column blobs), re-OCR each block, and merge the results back in
reading order. Because covered ink is erased before detection, recovered crops
can't duplicate captured text, so precision is preserved with no dedup. A block's
geometry thresholds derive from the page's own column width, so it adapts across
DPIs and column counts.

- **Gated / do-no-harm** — skips a page unless a meaningful fraction of its ink
  (default 10%) lies outside every detected box, so already-covered pages are a
  no-op.
- Recovered regions carry `engine="residual"` for auditing.
- Works with region recognizers and line recognizers that expose
  `recognize_region` (e.g. Tesseract).

On [NewsBench](https://github.com/nealcaren/newsbench) with PaddleX + GLM-OCR,
over pages with complete gold it lifts mean bowF1 0.961 → 0.980 and CER 0.919 →
0.937 by recovering columns PaddleX missed. With the stronger DocLayout-YOLO
detector there is little left uncovered, so residual is a near-no-op — it matters
most for the dependency-light PaddleX path.

## Phase 3: Post-Processing

### Text Cleaning

Reconstructs continuous text from OCR'd lines:

- **Dehyphenation**: `"com-" + "plete"` → `"complete"` (when next line starts lowercase)
- **Line joining**: Continuation lines joined with spaces
- **Paragraph breaks**: Detected via vertical gaps, column shifts, or terminal punctuation + uppercase
- **Semantic dashes preserved**: Em-dashes and spaced dashes kept intact

Disable with `text_cleaning=False` or `--no-text-cleaning`.

### Spell Correction

Optional SymSpell-based correction (`spell_check=True`):

- Corrects words not found in dictionary (edit distance ≤ 2)
- Preserves capitalization, punctuation, numbers, abbreviations
- Supports custom frequency dictionaries for corpus-specific vocabulary
- Logs all corrections for review

```python
pipe = Pipeline(spell_check=True)

# With corpus-specific dictionary
from newspaper_ocr.spell_checker import SpellChecker
checker = SpellChecker(dictionary_path="my_newspaper_words.txt")
```

## Output Formats

| Format | Flag | Content |
|--------|------|---------|
| `text` | `--output text` | Plain text, paragraphs separated by blank lines |
| `json` | `--output json` | Structured: regions, lines, bounding boxes, confidence, status |
| `hocr` | `--output hocr` | HTML with spatial coordinates (for text overlay on images) |
| `viewer` | `--output viewer` | OpenSeadragon review page: the scan, clickable region boxes, synced text pane (also registered as `html`) |

### JSON schema

The JSON formatter is the stable contract for downstream passes (review sites,
article segmentation, LLM enrichment). Each region carries:

| Field | Meaning |
|-------|---------|
| `id` | Stable per-page handle, `r0`, `r1`, ... in reading order |
| `label` | Region class from the detector (`text`, `title`, ...) |
| `bbox` | `x0`, `y0`, `x1`, `y1` in page pixels |
| `text` | Recognized text |
| `status` | `ok`, `timeout`, `repetition`, `error`, or `chunked_partial` |
| `confidence` | Detection confidence |
| `lines` | Per-line `text` / `confidence` / `bbox`, when the recognizer is line-level |

`status` is how a caller finds regions worth re-OCRing without re-reading the
images: `timeout` means the recognizer hit its wall-clock budget (the text is
the placeholder `[OCR timeout]`), `repetition` means the model looped and the
text was truncated, `error` means recognition raised, and `chunked_partial`
means a tall region was split into bands and at least one band still failed, so
the merged text is real but incomplete.

## Review Site

Checking a run means looking at the scan next to the text. `ReviewSite` writes a
static site that does that: one OpenSeadragon page per scan, an index over the
issue, and a `manifest.json` with every region.

```python
from newspaper_ocr import Pipeline
from newspaper_ocr.pdf import page_images
from newspaper_ocr.viewer import ReviewSite

pipe = Pipeline(recognizer="glm-ocr")
site = ReviewSite("site/industrial-worker-1912-05-01", title="Industrial Worker")

for image in page_images("issue.pdf", rotate=90):
    site.add_page(pipe.analyze(image))

site.write()
```

`Pipeline.analyze` is `run` without the formatting step — it returns the
`PageLayout`, so a page can be written to the site and to JSON without OCRing it
twice.

On a page: click a region on the scan and its text scrolls into view; click a
paragraph and the viewer zooms to its box. Regions are tinted by `status`, so
timeouts and truncated loops stand out instead of hiding in the JSON. Dragging
pans as usual — only a click selects.

`manifest.json` is the machine-readable half, carrying each page's dimensions and
every region's `id`, `label`, `bbox`, `text`, `status` and `confidence`, plus
per-page and whole-issue status counts. A re-OCR pass can find the regions worth
redoing from it without touching the images.

Pages load OpenSeadragon from a CDN, so the site needs network access to work.
For an offline or archival copy, drop `openseadragon.min.js` (and its `images/`
sprite directory) into the output and pass `openseadragon_url=`. Page scans are
written to `scans/` precisely so they don't collide with those sprites.

## Architecture

Every stage is a swappable component behind an abstract interface. Adding a new backend = one file + one registry entry.

```python
# Custom detector
from newspaper_ocr.detectors.base import Detector
class MyDetector(Detector):
    def detect(self, image) -> PageLayout: ...

# Custom recognizer
from newspaper_ocr.recognizers.base import LineRecognizer
class MyRecognizer(LineRecognizer):
    def recognize(self, line) -> Line: ...

# Plug into pipeline
pipe = Pipeline(detector=MyDetector(), recognizer=MyRecognizer())
```
