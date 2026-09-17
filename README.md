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
historical newspaper pages — **the detector matters more than the recognizer.**
Every combination of the two strong detectors × three recognizers (n = 15,
`overall` = 1 − CER, order-sensitive; higher is better):

| Detector | Recognizer | overall | cased | bowF1 |
|:---|:---|:---:|:---:|:---:|
| **DocLayout-YOLO** | PaddleOCR-VL | **0.970** | 0.953 | 0.985 |
| **DocLayout-YOLO** | GLM-OCR | 0.959 | 0.943 | 0.985 |
| **DocLayout-YOLO** | Tesseract | 0.919 | 0.889 | 0.910 |
| AS-YOLO | PaddleOCR-VL | 0.816 | 0.801 | 0.937 |
| AS-YOLO | GLM-OCR | 0.803 | 0.788 | 0.942 |
| AS-YOLO | Tesseract | 0.620 | 0.602 | 0.706 |

Holding the recognizer fixed and only swapping the detector moves the score more
than anything else (+0.15 for the VLMs, **+0.30** for Tesseract). Practical guidance:

- **Best accuracy:** `detector="auto"` (→ DocLayout-YOLO) + a region VLM (`glm-ocr` or `paddleocr-vl`).
- **Free / fully local / no GPU:** DocLayout-YOLO + Tesseract still reaches 0.919.
- **Avoid** the whole-page (no-detector) path on dense pages — layout is the bottleneck.

## Architecture

Every stage is a swappable component behind an abstract interface (detector,
recognizer, formatter). Adding a backend is one file plus one registry entry —
see [docs/guide.md](docs/guide.md#architecture).

## License

MIT
