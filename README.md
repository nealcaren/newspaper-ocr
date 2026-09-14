# newspaper-ocr

Modular OCR pipeline for historical newspaper scans. Three-phase architecture with swappable backends at every stage.

## Pipeline

```
           Phase 1               Phase 2              Phase 3
           LAYOUT                OCR                  POST-PROCESSING

          ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
          │ Detection        │   │ Recognition      │   │ Region Repair    │
Image ──→ │ (AS YOLO or      │──→│ (Tesseract,      │──→│ (text dedup,     │──→ Output
JP2/JPG/  │  PP-DocLayout)   │   │  Kraken, TrOCR,  │   │  container split)│    text
PNG       │                  │   │  LightOnOCR,     │   │                  │    json
          │ Layout Proc.     │   │  GLM-OCR,        │   │ Text Cleaning    │    hOCR
          │ (reading order,  │   │  tesserocr,      │   │ (dehyphenation)  │
          │  dedup, merge)   │   │  EffOCR)         │   │                  │
          │                  │   │                  │   │ Spell Check      │
          └──────────────────┘   └──────────────────┘   └──────────────────┘
```

**Phase 1 — Layout:** Detect regions (articles, headlines, ads) and text lines. Reorder into newspaper reading order (columns left-to-right, top-to-bottom). Deduplicate overlapping detections, fill gaps.

**Phase 2 — OCR:** Recognize text in each detected line or region. Swappable backends with different speed/accuracy tradeoffs.

**Phase 3 — Post-Processing:** Repair regions now that their text is known — drop provable duplicates, split double-detected columns, merge ad fragments. Reconstruct continuous text from OCR'd lines. Rejoin hyphenated words across line breaks. Join continuation lines into paragraphs. Optional spell correction.

## Installation

```bash
pip install newspaper-ocr

# Tesseract (requires system install):
#   macOS: brew install tesseract
#   Ubuntu: apt install tesseract-ocr

# Optional backends:
pip install "newspaper-ocr[kraken]"       # Kraken OCR (fast, GPU optional)
pip install "newspaper-ocr[trocr]"       # TrOCR (fine-tuned, GPU recommended)
pip install "newspaper-ocr[lightonocr]"  # LightOnOCR (best accuracy, GPU required)
pip install "newspaper-ocr[glm-ocr]"     # GLM-OCR vision-language model
pip install "newspaper-ocr[paddlex]"      # PP-DocLayout detector

# EfficientOCR (installed separately from fork):
pip install git+https://github.com/nealcaren/efficient_ocr.git
```

## Quick Start

### Python

```python
from newspaper_ocr import Pipeline

# Defaults: AS YOLO detection + Tesseract recognition
pipe = Pipeline()
text = pipe.ocr("page.jp2")

# Fast mode (tesserocr C API, ~4x faster)
pipe = Pipeline(recognizer="tesserocr")

# With spell correction
pipe = Pipeline(recognizer="tesserocr", spell_check=True)

# JSON output with bounding boxes and confidence scores
pipe = Pipeline(output="json")
result = pipe.ocr("page.jp2")

# Bundled fine-tuned model for historical newspapers (recommended)
pipe = Pipeline(recognizer="tesseract", recognizer_model="news_combo_fast")

# Disable layout post-processing (for non-newspaper documents)
pipe = Pipeline(layout_processing=False)

# Batch processing
results = pipe.ocr_batch(["page1.jp2", "page2.jp2", "page3.jp2"])
```

### Command Line

```bash
# Basic OCR
newspaper-ocr page.jp2

# Fast mode with JSON output
newspaper-ocr page.jp2 --backend tesserocr --output json

# With spell correction
newspaper-ocr page.jp2 --backend tesserocr --spell-check

# Batch processing to files
newspaper-ocr *.jp2 --outdir results/ --output text

# Fine-tuned model (bundled, recommended)
newspaper-ocr page.jp2 --model news_combo_fast

# Disable post-processing
newspaper-ocr page.jp2 --no-layout-processing --no-text-cleaning
```

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

Two detection backends, plus battle-tested newspaper layout post-processing.

### Detectors

| Detector | What it finds | Speed | Best for |
|----------|--------------|-------|----------|
| `as_yolo` (default) | Regions + lines | ~8s/page | Line-level OCR (Tesseract, EffOCR) |
| `paddlex` | Regions only (20 categories) | varies | Region-level OCR, detailed layout analysis |

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

Three recognition backends with different speed/accuracy tradeoffs.

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
| `effocr` | line | ~50s | 11.2% | Contrastive char/word matching, ONNX |

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
        timeout=25,              # per-region wall-clock budget, both modes
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
than silently empty text.

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

## Phase 3: Post-Processing

### Region Repair

Layout processing runs before OCR, so it only ever sees geometry. On dense 6–9
column broadsheets that leaves one defect standing: the detector emits both a
tall full-column region **and** the paragraph regions inside it. Tall-narrow
against wide-short, the two boxes are not near-duplicates and neither contains
the other, so IoU dedup keeps both — and OCR then reads the same passage twice,
at two different qualities, into the page JSON, full-text search, and every
article that cites those region ids.

`RegionRepair` runs after recognition, where the text is available to prove what
is redundant:

```python
pipe = Pipeline(recognizer="glm-ocr", region_repair=True)
```

```bash
newspaper-ocr page.jp2 --backend glm-ocr --region-repair
```

Off by default: it spends extra recognizer calls re-reading crops, and it is
dense multi-column pages that need it.

1. **Lossless text dedup** — a region is dropped only when another provably
   already carries its text: identical text in an overlapping box (the better
   read stays — higher `status`, then longer text), or text that is a strict
   substring of an overlapping region's text. Never on fuzzy or token-overlap
   similarity. A column read and its paragraph reads share most of their tokens
   while each holds OCR-variants the other lacks, so dropping on similarity
   deletes text no other region has.
2. **Container split** — when smaller regions sharing a column cover most of a
   region's height, that region is a duplicate container read. Dropping it would
   lose whatever the paragraph reads missed *between* them, so instead the
   uncovered vertical strips are re-OCR'd as their own regions and the container
   goes: covered text survives through the cleaner inner reads, uncovered text
   survives as new strips. Fully covered containers are dropped without a model
   call. A container whose strip re-OCR fails is kept whole.
3. **Fragmented-ad merge** — clusters of overlapping fragments of one display ad
   are unioned and re-OCR'd once, so the ad reads as prose instead of shards.
   Two guards keep the merge from *creating* duplicate text, and both are
   load-bearing: a union that would engulf a region outside the cluster is
   abandoned (re-OCR would copy that neighbour's words into the merge while the
   neighbour keeps them), as is a union spanning most of the page. Nested pairs
   — a container and its contents — are never clustered; that is pass 2's job.

New regions carry ids derived from what they came from (`r7` splits into
`r7s0`, `r7s1`; a merge of `r3` and `r5` becomes `r3m`), so references into a
page stay readable instead of being renumbered out from under downstream work.

Repair is **non-destructive**. The first call snapshots the recognized regions
into `PageLayout.raw_regions` and every call recomputes `regions` from that
snapshot, so it is idempotent, re-runnable with different thresholds, and cannot
destroy the raw OCR layer:

```python
from newspaper_ocr.region_repair import RegionRepair
from newspaper_ocr.recognizers.base import recognize_crop

layout = pipe.analyze(image)
repair = RegionRepair(container_coverage=0.75)
repair.repair(layout, lambda crop: recognize_crop(pipe.recognizer, crop))
print(repair.last_report)   # what changed, per pass
```

Both re-OCR passes need that callback. Without one — a line-level recognizer
with no region fallback, say — they are skipped rather than approximated: the
container stays whole and the fragments stay separate. Stitching together text
nobody read is the one outcome worse than a duplicate.

Every threshold is a constructor argument; see `RegionRepair`'s docstring for
what each trades off. The defaults are the ones validated end-to-end on *The
Negro World* (1921–1933).

### Duplicate Page Scans

Microfilm and vendor PDFs routinely carry the same physical page twice as two
different scans — a "+2 offset" rescan, a front page shot three times, a whole
second section reprinted. The crops and contrast differ, so the bytes and the
md5 differ, and hash-based dedup never sees them. Their text does:

```python
from newspaper_ocr.region_repair import find_duplicate_pages

pages = [pipe.analyze(image) for image in page_images("issue.pdf")]
for dup in find_duplicate_pages(pages):
    print(f"page {dup.index} repeats page {dup.duplicate_of} ({dup.similarity:.2f})")
```

The decision uses `difflib.SequenceMatcher.ratio()`. Across 60 clean issues,
true duplicate scans score 0.47–0.96 and distinct newspaper pages 0.06 or below,
with nothing legitimate in between — hence the 0.35 default threshold; the gap,
not the exact value, is what makes it safe. `quick_ratio()` is the trap: as a
frequency-based upper bound it scores *all* newspaper pages 0.7–0.95, because
any two pages of English prose use the same letters in similar proportions. It
is used only as a pre-filter to skip comparisons that cannot reach the
threshold.

Nothing is modified — which copy of a duplicated page to keep is the caller's
call, and the later scan is often the better one.

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
| `id` | Stable per-page handle, `r0`, `r1`, ... in reading order (region repair derives `r0s0`, `r0m` for regions it creates) |
| `label` | Region class from the detector (`text`, `title`, ...) |
| `bbox` | `x0`, `y0`, `x1`, `y1` in page pixels |
| `text` | Recognized text |
| `status` | `ok`, `timeout`, `repetition`, or `error` |
| `confidence` | Detection confidence |
| `lines` | Per-line `text` / `confidence` / `bbox`, when the recognizer is line-level |

`status` is how a caller finds regions worth re-OCRing without re-reading the
images: `timeout` means the recognizer hit its wall-clock budget (the text is
the placeholder `[OCR timeout]`), `repetition` means the model looped and the
text was truncated, and `error` means recognition raised.

`text` is always a string. See [Recognizer return contract](#recognizer-return-contract)
for why that is worth stating.

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

### Recognizer return contract

`RegionRecognizer.recognize(region)` returns the **`Region`** — not a
`(text, status)` tuple, not a bare string. `region.text` is always a `str`
(`""` when nothing was recognized) and `region.status` is one of
`REGION_STATUSES`. Implementations may mutate the region they were given and
return it, which is what the bundled recognizers do.

The `str` half matters more than it looks: a recognizer that returns
`(text, status)` and a caller that assigns the whole tuple to `region.text`
produce JSON with list-typed `"text"`, which breaks every downstream consumer
that expects a string — quietly, one page at a time. Code that re-OCRs a bare
crop rather than a detected region should go through `recognize_crop`, which
normalizes any of those shapes back to `(str, str)`:

```python
from newspaper_ocr.recognizers.base import recognize_crop

text, status = recognize_crop(pipe.recognizer, page_image.crop(box))
```

## License

MIT
