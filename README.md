# newspaper-ocr

Modular OCR pipeline for historical newspaper scans. Combines state-of-the-art layout detection with swappable recognition backends.

## Features

- **Two detection backends**: AS YOLO (fast, line-level) and PP-DocLayout (region-level, 20 layout categories)
- **Three recognition backends**: Tesseract, tesserocr (fast C API), EfficientOCR
- **Newspaper-aware layout processing**: Column-aware reading order, region deduplication, gap filling
- **Multiple output formats**: Plain text, structured JSON, hOCR
- **Works on any resolution**: From full-res LOC JP2 scans to reduced-resolution images

## Installation

```bash
pip install newspaper-ocr

# With Tesseract (requires system tesseract):
# macOS: brew install tesseract
# Ubuntu: apt install tesseract-ocr

# With fast C API bindings:
pip install newspaper-ocr[tesserocr]

# With EfficientOCR:
pip install newspaper-ocr[effocr]
```

## Quick Start

### Python

```python
from newspaper_ocr import Pipeline

# Simple — defaults (AS YOLO detection + Tesseract recognition)
pipe = Pipeline()
text = pipe.ocr("newspaper_page.jp2")
print(text)

# Fast mode (tesserocr + region-level recognition)
pipe = Pipeline(recognizer="tesserocr")
text = pipe.ocr("newspaper_page.jp2")

# JSON output with bounding boxes
pipe = Pipeline(output="json")
result = pipe.ocr("newspaper_page.jp2")

# Use a fine-tuned Tesseract model
pipe = Pipeline(
    recognizer="tesseract",
    recognizer_model="news_gold_v2.traineddata",
)

# Disable layout post-processing (for non-newspaper documents)
pipe = Pipeline(layout_processing=False)
```

### Command Line

```bash
# Basic OCR
newspaper-ocr page.jp2

# Fast mode with JSON output
newspaper-ocr page.jp2 --backend tesserocr --output json

# Batch processing
newspaper-ocr *.jp2 --outdir results/ --output text

# Use fine-tuned model
newspaper-ocr page.jp2 --model news_gold_v2.traineddata
```

## Recognition Backends

| Backend | Speed (1 page) | CER | Requirements |
|---------|----------------|-----|-------------|
| `tesseract` (line) | ~106s | 3.2%* | System tesseract |
| `tesseract` (region) | ~38s | TBD | System tesseract |
| `tesserocr` (region) | ~25s | TBD | tesserocr package |
| `effocr` | ~8s | 11.2% | efficient-ocr package |

*With fine-tuned `news_gold_v2` model. Baseline Tesseract is ~8.2% CER.

## Pipeline Architecture

```
Image → Detector → LayoutProcessor → Recognizer → Formatter → Output
         (YOLO)    (reading order)   (Tesseract)   (text/json)
```

Each stage is swappable. See the [design spec](docs/design.md) for details.

## License

MIT
