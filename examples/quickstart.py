"""Quick start: OCR a historical newspaper page with the recommended setups.

Two recommended configurations, best first:

1. Best accuracy (GPU) — the recovery ladder: DocLayout-YOLO layout detection,
   GLM-OCR region OCR, tall-region splitting, and a PaddleOCR-VL fallback for
   anything the primary can't read. Skipped automatically if the deps or a GPU
   aren't available.
2. Best without a GPU — Tesseract with the bundled `news_combo_fast` model,
   fine-tuned on pre-1930 newspaper text (2.9% CER vs 9.0% for stock `eng`).

For the full, annotated walkthrough (with rendered results, a review site, and
the status contract) see examples/newspaper_ocr_walkthrough.ipynb.

Usage:
    cd newspaper-ocr
    python examples/quickstart.py
"""

import time
from pathlib import Path

# 2800px reduced-resolution scan (a typical working resolution).
SAMPLE_PAGE = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_page_2800.jpg"


def show(text, elapsed=None):
    if elapsed is not None:
        print(f"Time: {elapsed:.1f}s | {len(text)} chars")
    print(text[:800].rstrip())
    print("...\n")


def main():
    from newspaper_ocr import Pipeline

    if not SAMPLE_PAGE.exists():
        print(f"Sample page not found: {SAMPLE_PAGE} (run from the repo root).")
        return

    print(f"Sample page: {SAMPLE_PAGE.name}\n")

    # --- 1. Best accuracy: the GPU recovery ladder --------------------------
    # DocLayout-YOLO -> GLM-OCR (primary) -> chunk tall regions -> PaddleOCR-VL
    # (fallback, do-no-harm). Needs a GPU and:
    #   pip install "newspaper-ocr[doclayout,glm-ocr,paddleocr-vl]"
    # PaddleOCR-VL needs an A100-class card. Skips cleanly if unavailable.
    print("=" * 64)
    print("Recommended (GPU): DocLayout-YOLO + GLM-OCR + PaddleOCR-VL fallback")
    print("=" * 64)
    try:
        from collections import Counter

        import torch
        from PIL import Image

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU (VLM OCR on CPU is impractically slow)")

        from newspaper_ocr.recognizers.glm_ocr import GlmOcrRecognizer
        from newspaper_ocr.recognizers.paddleocr_vl import PaddleOcrVlRecognizer

        pipe = Pipeline(
            detector="doclayout_yolo",
            recognizer=GlmOcrRecognizer(mode="local"),   # primary
            fallback=PaddleOcrVlRecognizer(),            # backup for failures
            chunk_tall_regions=True,                     # split tall timeouts
            text_cleaning=False,                         # VLM text is already clean
        )
        t0 = time.time()
        layout = pipe.analyze(Image.open(SAMPLE_PAGE).convert("RGB"))
        elapsed = time.time() - t0
        print(f"Time: {elapsed:.1f}s | {len(layout.regions)} regions")
        print("status:", dict(Counter(r.status for r in layout.regions)))
        recovered = [r for r in layout.regions if r.engine]
        if recovered:
            print(f"recovered by fallback: {len(recovered)} "
                  f"({dict(Counter(r.engine for r in recovered))})")
        print()
        print(layout.text[:800].rstrip())
        print("...\n")
    except Exception as e:
        print(f"Skipped (needs a GPU + [doclayout,glm-ocr,paddleocr-vl]): {e}\n")

    # --- 2. Best without a GPU: bundled fine-tuned Tesseract ----------------
    print("=" * 64)
    print("Recommended (CPU): Tesseract + bundled news_combo_fast model")
    print("=" * 64)
    try:
        pipe = Pipeline(recognizer="tesseract", recognizer_model="news_combo_fast")
        t0 = time.time()
        text = pipe.ocr(str(SAMPLE_PAGE))
        show(text, time.time() - t0)
    except Exception as e:
        print(f"Skipped (needs Tesseract installed): {e}\n")

    # --- 3. JSON output (the downstream contract) --------------------------
    print("=" * 64)
    print("JSON output (first region)")
    print("=" * 64)
    try:
        import json

        pipe = Pipeline(recognizer="tesseract", recognizer_model="news_combo_fast",
                        output="json")
        data = json.loads(pipe.ocr(str(SAMPLE_PAGE)))
        print(f"Regions: {len(data['regions'])}")
        if data["regions"]:
            r = data["regions"][0]
            print(f"First region: id={r['id']} label={r['label']} "
                  f"status={r['status']} lines={len(r['lines'])}")
            print(f"Text: {r['text'][:200]}...")
    except Exception as e:
        print(f"Error: {e}")

    print("\nDone. For the full walkthrough with rendered results:")
    print("  examples/newspaper_ocr_walkthrough.ipynb")


if __name__ == "__main__":
    main()
