# newspaper-ocr

Modular OCR pipeline for historical newspaper scans. Three-phase architecture with swappable backends at every stage.

## Pipeline

```
           Phase 1                    Phase 2                    Phase 3
           LAYOUT                     OCR                        POST-PROCESSING

          ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
          │ Detection            │   │ Recognition          │   │ Text Cleaning        │
          │ (AS YOLO,            │   │ (Tesseract,          │   │ (dehyphenation,      │
Image ──→ │  DocLayout-YOLO,     │──→│  tesserocr, Kraken,  │──→│  line joining)       │──→ Output
JP2/JPG/  │  PP-DocLayout,       │   │  TrOCR, GLM-OCR,     │   │                      │    text
PNG/PDF   │  MinerU2.5;          │   │  LightOnOCR,         │   │ Spell Check          │    json
          │  + hole fill)        │   │  PaddleOCR-VL,       │   │ (SymSpell)           │    hOCR
          │ Layout Proc.         │   │  MinerU2.5, EffOCR,  │   │                      │    viewer
          │ (reading order,      │   │  hosted VLMs)        │   │                      │
          │  dedup, merge)       │   │                      │   │                      │
          └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

**Phase 1 — Layout:** Detect regions (articles, headlines, ads) and text lines. Reorder into newspaper reading order (columns left-to-right, top-to-bottom). Deduplicate overlapping detections, fill gaps.

**Phase 2 — OCR:** Recognize text in each detected line or region. Swappable backends with different speed/accuracy tradeoffs.

**Phase 3 — Post-Processing:** Reconstruct continuous text from OCR'd lines. Rejoin hyphenated words across line breaks. Join continuation lines into paragraphs. Optional spell correction.

## Installation

```bash
pip install newspaper-ocr

# Tesseract (requires system install):
#   macOS: brew install tesseract
#   Ubuntu: apt install tesseract-ocr

# Linux + CUDA GPU (best accuracy): MinerU2.5 + DocLayout-YOLO hole fill
pip install "newspaper-ocr[mineru,doclayout]"

# Mac / local (lowest memory): DocLayout-YOLO + GLM-OCR
pip install "newspaper-ocr[doclayout]"    # DocLayout-YOLO detector
pip install "newspaper-ocr[glm-ocr]"      # GLM-OCR vision-language model

# Other optional backends:
pip install "newspaper-ocr[paddlex]"      # PP-DocLayout detector
pip install "newspaper-ocr[paddleocr-vl]" # PaddleOCR-VL VLM
pip install "newspaper-ocr[kraken]"       # Kraken OCR (fast, GPU optional)
pip install "newspaper-ocr[trocr]"        # TrOCR (fine-tuned, GPU recommended)
pip install "newspaper-ocr[lightonocr]"   # LightOnOCR (GPU required)
pip install "newspaper-ocr[api]"          # OpenAI/OpenRouter or any OpenAI-compatible endpoint
pip install "newspaper-ocr[pdf]"          # multi-page PDF input

# EfficientOCR (installed separately from fork):
pip install git+https://github.com/nealcaren/efficient_ocr.git
```

## Quick Start

### Python

```python
from newspaper_ocr import Pipeline

# Default: auto detector (DocLayout-YOLO when installed) + Tesseract recognition
pipe = Pipeline()
text = pipe.ocr("page.jp2")

# Linux + CUDA GPU (best accuracy): MinerU2.5 reads the page, DocLayout-YOLO fills its holes
pipe = Pipeline(detector="mineru", hole_fill_detector="doclayout_yolo", recognizer="mineru")

# Mac / local (lowest memory): DocLayout-YOLO + GLM-OCR
pipe = Pipeline(recognizer="glm-ocr")     # detector="auto" -> doclayout_yolo

# Multi-page PDF: one result per page
pages = pipe.ocr_pdf("issue.pdf")

# Bundled fine-tuned Tesseract model (free, fully local)
pipe = Pipeline(recognizer="tesseract", recognizer_model="news_combo_fast")

# JSON output with bounding boxes, confidence, and status
pipe = Pipeline(output="json")

# Batch processing
results = pipe.ocr_batch(["page1.jp2", "page2.jp2", "page3.jp2"])
```

### Command Line

```bash
newspaper-ocr page.jp2                                     # basic OCR
newspaper-ocr page.jp2 --backend glm-ocr --output json    # Mac/local: DocLayout + GLM-OCR, JSON
newspaper-ocr page.jp2 --detector mineru --hole-fill-detector doclayout_yolo \
    --backend mineru                                       # Linux + CUDA: best accuracy
newspaper-ocr page.jp2 --model news_combo_fast            # bundled fine-tuned model
newspaper-ocr *.jp2 --outdir results/ --output text       # batch to files
newspaper-ocr issue.pdf --outdir results/                 # multi-page PDF, one file per page
newspaper-ocr page.jp2 --backend mineru --markup raw      # keep VLM HTML tables/LaTeX (default: plain text)
```

See the **[detailed guide](docs/guide.md)** for PDF input, every detector and
recognizer, the recovery ladder, the residual second pass, output formats
(incl. the JSON schema and review site), and the extension API.

## Which backend? (NewsBench results)

On [NewsBench](https://github.com/nealcaren/newsbench) — dense, multi-column
historical newspaper pages, n = 15. `overall` = 1 − CER (order-sensitive);
`cased` keeps case + punctuation; `bowF1` is order-free bag-of-words F1; `$/100pg`
is real API spend ($0 = local). Higher is better.

| Detector | Recognizer | newspaper-ocr | overall | cased | bowF1 | $/100pg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **MinerU2.5 + DocLayout holes** | MinerU2.5 | 0.9.0 | **0.974** | **0.957** | 0.985 | $0.00 |
| **DocLayout-YOLO** | PaddleOCR-VL | 0.8.1 | 0.970 | 0.953 | 0.985 | $0.00 |
| MinerU2.5 | MinerU2.5 | 0.9.0 | 0.966 | 0.950 | 0.981 | $0.00 |
| **DocLayout-YOLO** | GLM-OCR | 0.8.1 | 0.959 | 0.943 | 0.985 | $0.00 |
| PaddleX | GLM-OCR | 0.7.0 | 0.937 | 0.922 | 0.980 | $0.00 |
| PaddleX | Gemini-flash-lite | 0.7.0 | 0.936 | 0.915 | 0.944 | $4.51 |
| **DocLayout-YOLO** | Tesseract | 0.8.1 | 0.919 | 0.889 | 0.910 | $0.00 |
| PaddleX | GLM-OCR | 0.6.0 | 0.919 | 0.905 | 0.961 | $0.00 |
| PaddleX | Tesseract | 0.7.0 | 0.899 | 0.874 | 0.891 | $0.00 |
| none (whole page) | Gemini-flash-lite | — | 0.820 | 0.803 | 0.867 | $2.88 |
| AS-YOLO | PaddleOCR-VL | 0.8.1 | 0.816 | 0.801 | 0.937 | $0.00 |
| AS-YOLO | GLM-OCR | 0.8.1 | 0.803 | 0.788 | 0.942 | $0.00 |
| none (whole page) | Tesseract | — | 0.677 | 0.662 | 0.844 | $0.00 |
| AS-YOLO | Tesseract | 0.8.1 | 0.620 | 0.602 | 0.706 | $0.00 |

**The harness earns its keep.** Compare the same recognizer with no detector (raw
whole page) vs the full detect → layout pipeline: Tesseract **0.677 → 0.919
(+0.24)** and Gemini-flash-lite **0.820 → 0.936 (+0.12)**. The whole-page rows are
the no-harness baseline — everything above them is what layout detection buys.

**The detector matters more than the recognizer.** Holding the recognizer fixed
and only swapping the detector moves the score more than anything else (+0.15 for
the VLMs, **+0.30** for Tesseract). The version column also shows the residual
pass earning its keep: on PaddleX, turning it on by default (**0.6.0 → 0.7.0**)
lifted GLM-OCR from 0.919 → 0.937 by recovering columns PaddleX missed. Under
DocLayout-YOLO it's a near-no-op — the better detector leaves little uncovered — so
DocLayout needs no residual to reach the top.

Practical guidance:

- **Linux + CUDA GPU (best accuracy):** MinerU2.5 with DocLayout-YOLO filling its holes (0.974) — see [below](#best-result-mineru--doclayout-hole-fill-0974-local-free).
- **Mac / local (lowest memory):** `detector="auto"` (→ DocLayout-YOLO) + `glm-ocr` (0.959). On a CUDA box without MinerU, `paddleocr-vl` scores a bit higher (0.970).
- **Cheapest hosted:** PaddleX + Gemini-flash-lite reaches 0.936 at ~$4.51/100 pages.
- **Free / fully local / no GPU:** DocLayout-YOLO + Tesseract still reaches 0.919.
- **Avoid** the whole-page (no-detector) path on dense pages — layout is the bottleneck.

_(0.8.1 and 0.9.0 rows are the GPU matrix (L40S); 0.6.0/0.7.0 rows are the
earlier Mac/MLX + hosted-API runs. Full sheet with tokens/speed:_
`python scoresheet.py` _in the [NewsBench](https://github.com/nealcaren/newsbench) repo.)_

## Leaderboard OCR models on NewsBench

Models that top general document-parsing leaderboards (e.g. OmniDocBench) do **not**
automatically top NewsBench — dense, multi-column newspaper pages are a layout and
reading-order problem, not just a character-recognition one. The dividing line is
whether a system does layout analysis at all.

**Hosted VLMs: the harness makes them; running whole-page breaks them.** The same
model, given the raw full page in one call vs. run through our detect → region →
reading-order harness (DocLayout-YOLO), n = 15:

| Recognizer | whole page | + our harness | Δ | $/100pg |
|:---|:---:|:---:|:---:|:---:|
| gpt-5.6-luna | 0.503 | **0.975** | +0.47 | $4.68 |
| gemini-3.5-flash-lite | 0.820 | 0.958 | +0.14 | $2.41 |
| mistral-small-2603 | 0.202 | 0.958 | +0.76 | $0.82 |
| deepseek-v4.1-flash | — | 0.957 | — | $3.47 |
| glm-5.3-flash | — | 0.938 | — | $0.76 |
| gemma-3-27b-it | 0.226 | 0.921 | +0.70 | $0.43 |

Every model gains massively from the harness; the whole-page column is where
capable VLMs go to fail on multi-column layouts.

**End-to-end document models, whole-page** (no external harness), n = 15:

| Model | overall | what it is |
|:---|:---:|:---|
| **MinerU2.5-1.2B** | **0.966** | full parsing **pipeline** — does its own layout + reading order |
| dots.ocr (~3B) | 0.533 | bare OCR VLM |
| OvisOCR2 (0.8B) | −0.230 | bare OCR VLM (over-generates ~2×) |
| TeleOCR (~7B) | −0.322 | bare OCR VLM (repetition collapse) |

The bare OCR VLMs — however high they rank on clean-document benchmarks — collapse
on newspapers (0.53 down to *negative*, where per-page edit distance exceeds the
gold length). **MinerU2.5 is the exception because it is a pipeline**: it runs its
own layout and reading-order stage internally, so it reaches 0.966 — essentially
tied with our harness (0.970). The lesson isn't "our recognizer wins," it's
**layout handling is the whole game**: a raw model needs a layout pipeline — ours,
or one built in — to read a full page. Our harness supplies that layer to *any*
recognizer, from free Tesseract (0.919) to a hosted VLM (0.975).

### Best result: MinerU + DocLayout hole fill (0.974, local, free)

MinerU2.5 has the best reading order (gap 0.015) but drops ~2.7% of words in
localized holes its layout misses; DocLayout-YOLO *covers* those holes. Since
0.9.0 this combination is built in: **MinerU supplies the base regions and their
reading order, DocLayout adds only the inked boxes MinerU missed, and MinerU reads
everything**:

```bash
pip install "newspaper-ocr[mineru,doclayout]"
newspaper-ocr page.jpg --detector mineru --hole-fill-detector doclayout_yolo --backend mineru
```

```python
Pipeline(detector="mineru", hole_fill_detector="doclayout_yolo", recognizer="mineru")
```

n = 15:

| System | overall | cased | bowF1 | miss% | $/100pg |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MinerU + DocLayout holes** (0.9.0) | **0.974** | **0.957** | 0.985 | 1.6 | **$0.00** |
| MinerU2.5 alone (0.9.0) | 0.966 | 0.950 | 0.981 | 2.7 | $0.00 |
| DocLayout + PaddleOCR-VL (0.8.1) | 0.970 | 0.953 | 0.985 | 1.8 | $0.00 |
| DocLayout + gpt-5.6-luna (hosted) | 0.975 | 0.951 | 0.951 | 3.6 | $4.68 |

Hole fill is do-no-harm in practice: most pages get no holes and come through
byte-identical to MinerU alone, while pages MinerU partly missed gain the most
(one page goes 0.890 → 0.990). Duplicated and novel text stay flat. It matches the
best *hosted* result while being **fully local and free**. (The out-of-library
prototype scored 0.975; the 0.001 difference is one story-continuation block whose
correct position the layout alone can't reveal.)

Holes are recognized with MinerU because they are isolated blocks in its comfort
zone; feeding it DocLayout's coarse *columns* makes it duplicate text, so without
MinerU's own boxes, use a region-native recognizer (GLM-OCR / PaddleOCR-VL). Any
two detectors can be combined this way (`hole_fill_detector=`); see the
[guide](docs/guide.md#combining-detectors-hole-fill). MinerU needs a CUDA GPU;
on a Mac it is impractically slow.

**On vLLM it is ~20× faster at the same accuracy.** `pip install
"newspaper-ocr[mineru,mineru-vllm,doclayout]"` and use the `mineru-vllm` detector and
recognizer: ~8 s/page on an L40S instead of ~80 s (and no 30-minute repetition
loops), scoring 0.973 on NewsBench vs 0.974. See
[Running at scale](docs/guide.md#running-at-scale) for batch runs and Slurm.

```bash
newspaper-ocr issue.pdf --detector mineru-vllm --hole-fill-detector doclayout_yolo \
    --backend mineru-vllm --outdir results/
```

See [docs/error-analysis.md](docs/error-analysis.md) for where the top configs
still miss or over-transcribe, cropped from the scans.

## Architecture

Every stage is a swappable component behind an abstract interface (detector,
recognizer, formatter). Adding a backend is one file plus one registry entry —
see [docs/guide.md](docs/guide.md#architecture).

## License

MIT
