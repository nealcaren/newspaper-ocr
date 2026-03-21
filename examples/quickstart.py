"""Quick start example: OCR a historical newspaper page.

This example uses the sample page bundled in tests/fixtures/:
- sn84020143 / 1905-12-17 / seq-1 (St. Louis Palladium, front page)

Runs three backends and compares output.

Usage:
    cd newspaper-ocr
    source .venv/bin/activate
    python examples/quickstart.py
"""

import time
from pathlib import Path

# Use the 2800px reduced-resolution image (typical working resolution)
SAMPLE_PAGE = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_page_2800.jpg"
SAMPLE_FULL = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_page_full.jpg"


def main():
    from newspaper_ocr import Pipeline

    if not SAMPLE_PAGE.exists():
        print(f"Sample page not found: {SAMPLE_PAGE}")
        print("Run from the newspaper-ocr repo root.")
        return

    print(f"Sample page: {SAMPLE_PAGE}")
    print(f"  sn84020143 / St. Louis Palladium / 1905-12-17 / front page")
    print()

    # --- Tesseract (region mode, fast) ---
    print("=" * 60)
    print("Backend: tesserocr (region mode)")
    print("=" * 60)
    try:
        pipe = Pipeline(recognizer="tesserocr", output="text")
        t0 = time.time()
        text = pipe.ocr(str(SAMPLE_PAGE))
        elapsed = time.time() - t0
        print(f"Time: {elapsed:.1f}s")
        print(f"Output: {len(text)} chars")
        print()
        print(text[:1000])
        print("...")
        print()
    except Exception as e:
        print(f"Error: {e}\n")

    # --- Tesseract (line mode, for comparison) ---
    print("=" * 60)
    print("Backend: tesseract (line mode)")
    print("=" * 60)
    try:
        pipe = Pipeline(recognizer="tesseract", output="text")
        t0 = time.time()
        text = pipe.ocr(str(SAMPLE_PAGE))
        elapsed = time.time() - t0
        print(f"Time: {elapsed:.1f}s")
        print(f"Output: {len(text)} chars")
        print()
        print(text[:1000])
        print("...")
        print()
    except Exception as e:
        print(f"Error: {e}\n")

    # --- EffOCR ---
    print("=" * 60)
    print("Backend: effocr")
    print("=" * 60)
    try:
        pipe = Pipeline(recognizer="effocr", output="text")
        t0 = time.time()
        text = pipe.ocr(str(SAMPLE_PAGE))
        elapsed = time.time() - t0
        print(f"Time: {elapsed:.1f}s")
        print(f"Output: {len(text)} chars")
        print()
        print(text[:1000])
        print("...")
        print()
    except Exception as e:
        print(f"Error: {e}\n")

    # --- JSON output example ---
    print("=" * 60)
    print("JSON output (first region)")
    print("=" * 60)
    try:
        import json
        pipe = Pipeline(recognizer="tesserocr", output="json")
        result = pipe.ocr(str(SAMPLE_PAGE))
        data = json.loads(result)
        print(f"Regions: {len(data['regions'])}")
        if data["regions"]:
            r = data["regions"][0]
            print(f"First region: {r['label']}, {len(r['lines'])} lines")
            print(f"Text: {r['text'][:200]}...")
    except Exception as e:
        print(f"Error: {e}")

    print()
    print("Done! Try it yourself:")
    print("  from newspaper_ocr import Pipeline")
    print('  pipe = Pipeline(recognizer="tesserocr")')
    print(f'  text = pipe.ocr("{SAMPLE_PAGE}")')


if __name__ == "__main__":
    main()
