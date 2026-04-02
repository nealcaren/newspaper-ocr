# newspaper-ocr

Modular OCR pipeline for historical newspaper scans. Three-phase architecture with swappable backends at every stage.

## Pipeline

```
           Phase 1               Phase 2              Phase 3
           LAYOUT                OCR                  POST-PROCESSING

          ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
          │ Detection        │   │ Recognition      │   │ Text Cleaning    │
Image ──→ │ (AS YOLO or      │──→│ (Tesseract,      │──→│ (dehyphenation,  │──→ Output
JP2/JPG/  │  PP-DocLayout)   │   │  Kraken,         │   │  line joining)   │    text
PNG       │                  │   │  tesserocr,      │   │                  │    json
          │ Layout Proc.     │   │  EffOCR)         │   │ Spell Check      │    hOCR
          │ (reading order,  │   │                  │   │ (SymSpell)       │
          │  dedup, merge)   │   │                  │   │                  │
          └──────────────────┘   └──────────────────┘   └──────────────────┘
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

# Optional backends:
pip install "newspaper-ocr[kraken]"       # Kraken OCR (fast, GPU optional)
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

Disable with `layout_processing=False`.

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

See [dangerouspress-ocr-finetune](https://github.com/nealcaren/ocr-finetune) for the training pipeline.

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
| `json` | `--output json` | Structured: regions, lines, bounding boxes, confidence |
| `hocr` | `--output hocr` | HTML with spatial coordinates (for text overlay on images) |

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

## License

MIT
