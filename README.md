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
JP2/JPG/  │  PP-DocLayout)       │   │  TrOCR, GLM-OCR,     │   │                      │    text
PNG/PDF   │                      │   │  LightOnOCR,         │   │ Spell Check          │    json
          │ Layout Proc.         │   │  PaddleOCR-VL,       │   │ (SymSpell)           │    hOCR
          │ (reading order,      │   │  EffOCR)             │   │                      │    viewer
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

# Recommended detector + a strong recognizer:
pip install "newspaper-ocr[doclayout]"    # DocLayout-YOLO detector (best layout)
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

# Recommended for accuracy: DocLayout-YOLO + a region VLM
pipe = Pipeline(recognizer="glm-ocr")     # detector="auto" -> doclayout_yolo

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
newspaper-ocr page.jp2 --backend glm-ocr --output json    # DocLayout + GLM-OCR, JSON
newspaper-ocr page.jp2 --model news_combo_fast            # bundled fine-tuned model
newspaper-ocr *.jp2 --outdir results/ --output text       # batch to files
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
| **DocLayout-YOLO** | PaddleOCR-VL | 0.8.1 | **0.970** | **0.953** | 0.985 | $0.00 |
| **DocLayout-YOLO** | GLM-OCR | 0.8.1 | 0.959 | 0.943 | 0.985 | $0.00 |
| PaddleX | GLM-OCR | 0.7.0 | 0.937 | 0.922 | 0.980 | $0.00 |
| PaddleX | Gemini-flash-lite | 0.7.0 | 0.936 | 0.915 | 0.944 | $4.51 |
| **DocLayout-YOLO** | Tesseract | 0.8.1 | 0.919 | 0.889 | 0.910 | $0.00 |
| PaddleX | GLM-OCR | 0.6.0 | 0.919 | 0.905 | 0.961 | $0.00 |
| PaddleX | Tesseract | 0.7.0 | 0.899 | 0.874 | 0.891 | $0.00 |
| none (whole page) | Gemini-flash-lite | 0.7.0 | 0.820 | 0.803 | 0.867 | $2.88 |
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
the VLMs, **+0.30** for Tesseract). Two things the version column captures:

- **The residual pass (0.6.0 → 0.7.0) rescues the weaker detector.** On PaddleX,
  turning it on by default lifted GLM-OCR from 0.919 → 0.937. Under DocLayout-YOLO
  it's a near-no-op — the better detector leaves little uncovered — so DocLayout
  needs no residual to reach the top.
- **0.8.1 fixed VLM local (GPU) mode** (empty output from a `device_map`/timeout
  bug); the DocLayout rows are that GPU matrix.

Practical guidance:

- **Best accuracy:** `detector="auto"` (→ DocLayout-YOLO) + a region VLM (`glm-ocr` or `paddleocr-vl`).
- **Cheapest hosted:** PaddleX + Gemini-flash-lite reaches 0.936 at ~$4.51/100 pages.
- **Free / fully local / no GPU:** DocLayout-YOLO + Tesseract still reaches 0.919.
- **Avoid** the whole-page (no-detector) path on dense pages — layout is the bottleneck.

_(0.8.1 rows are the GPU matrix, all one environment; 0.6.0/0.7.0 rows are the
earlier Mac/MLX + hosted-API runs. Full sheet with tokens/speed:_
`python scoresheet.py` _in the [NewsBench](https://github.com/nealcaren/newsbench) repo.)_

## Architecture

Every stage is a swappable component behind an abstract interface (detector,
recognizer, formatter). Adding a backend is one file plus one registry entry —
see [docs/guide.md](docs/guide.md#architecture).

## License

MIT
