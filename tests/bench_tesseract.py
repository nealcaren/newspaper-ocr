"""Benchmark Tesseract line vs region vs tesserocr modes."""
import time
from pathlib import Path
from PIL import Image
from newspaper_ocr.detectors.as_yolo import AsYoloDetector
from newspaper_ocr.recognizers.tesseract import TesseractRecognizer

# Use a real newspaper scan
img_path = "/Volumes/Lightning/chronicling-america/loc_downloads/sn84025908/1856-08-30/seq-4.jp2"
if not Path(img_path).exists():
    print(f"Image not found: {img_path}")
    print("Skipping benchmark.")
    raise SystemExit(0)

img = Image.open(img_path)

detector = AsYoloDetector()
layout = detector.detect(img)
total_lines = sum(len(r.lines) for r in layout.regions)
print(f"Detected {len(layout.regions)} regions, {total_lines} lines")

# Mode 1: Line-by-line subprocess (psm 7)
rec_line = TesseractRecognizer(mode="line")
t0 = time.time()
for region in layout.regions:
    for line in region.lines:
        rec_line.recognize(line)
t1 = time.time()
print(f"Line mode (psm 7): {t1-t0:.1f}s for {total_lines} lines")

# Mode 2: Region-level subprocess (psm 6)
rec_region = TesseractRecognizer(mode="region")
t0 = time.time()
for region in layout.regions:
    rec_region.recognize_region(region)
t2 = time.time()
print(f"Region mode (psm 6): {t2-t0:.1f}s for {len(layout.regions)} regions")

# Mode 3: tesserocr line mode
try:
    from newspaper_ocr.recognizers.tesserocr_backend import TesserocrRecognizer

    rec_capi_line = TesserocrRecognizer(mode="line")
    t0 = time.time()
    for region in layout.regions:
        for line in region.lines:
            rec_capi_line.recognize(line)
    t3 = time.time()
    print(f"tesserocr line (C API): {t3-t0:.1f}s for {total_lines} lines")

    # Mode 4: tesserocr region mode
    rec_capi_region = TesserocrRecognizer(mode="region")
    t0 = time.time()
    for region in layout.regions:
        rec_capi_region.recognize_region(region)
    t4 = time.time()
    print(f"tesserocr region (C API): {t4-t0:.1f}s for {len(layout.regions)} regions")
except ImportError:
    print("tesserocr not available, skipping C API benchmark")
