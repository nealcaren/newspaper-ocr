from __future__ import annotations

import threading

from PIL import Image

from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.models import BBox, PageLayout, Region

#: Published DocLayout-YOLO checkpoints, keyed by a short variant name.
#: Each entry is (hub repo, filename, native inference size).  The 1280 model
#: is a newer (Jan 2025) checkpoint trained at higher resolution; it is the
#: default here because newspaper pages are large and dense, and the extra
#: resolution recovers small regions (bylines, captions) the 1024 model misses.
VARIANTS: dict[str, tuple[str, str, int]] = {
    "docstructbench-1024": (
        "juliozhao/DocLayout-YOLO-DocStructBench",
        "doclayout_yolo_docstructbench_imgsz1024.pt",
        1024,
    ),
    "docstructbench-1280": (
        "juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501",
        "doclayout_yolo_docstructbench_imgsz1280_2501.pt",
        1280,
    ),
}
DEFAULT_VARIANT = "docstructbench-1280"


#: Serializes the torch.load override in :func:`_load_yolov10` so two concurrent
#: detector constructions can't race on saving/restoring the global.
_LOAD_LOCK = threading.Lock()


def _load_yolov10(yolov10_cls, model_path: str):
    """Load a DocLayout-YOLO checkpoint across PyTorch versions.

    PyTorch 2.6 flipped ``torch.load``'s default to ``weights_only=True``, which
    refuses to unpickle the ``YOLOv10DetectionModel`` object these checkpoints
    contain. doclayout-yolo's loader doesn't override that, so loading fails on
    modern torch. The checkpoints come from the official DocLayout-YOLO Hub
    repos, so we load them in full-pickle mode.

    ``torch.load`` is process-global, so the override is held under a lock and
    restored immediately, keeping concurrent detector constructions from racing
    on it. (Enumerating every checkpoint class for ``safe_globals`` instead would
    be brittle across doclayout-yolo versions.)
    """
    try:
        import torch
    except ImportError:
        return yolov10_cls(model_path)

    with _LOAD_LOCK:
        original_load = torch.load

        def _full_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = _full_load
        try:
            return yolov10_cls(model_path)
        finally:
            torch.load = original_load


class DocLayoutYoloDetector(Detector):
    """DocLayout-YOLO region detector (Zhao et al., arXiv:2410.12628).

    A YOLOv10-based document layout model. Region-only: it predicts bounding
    boxes (title, plain text, figure, table, ...) but no text lines, so pages go
    to region-level OCR the same way :class:`PaddleXDetector` does.

    Requires: ``pip install "newspaper-ocr[doclayout]"`` (the ``doclayout-yolo``
    package). The checkpoint is downloaded from the Hugging Face Hub and cached.
    """

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        model_repo: str | None = None,
        model_filename: str | None = None,
        imgsz: int | None = None,
        conf: float = 0.2,
        device: str | None = None,
        **kwargs,
    ):
        # model_dir / skip_lines are passed by Pipeline for every detector; this
        # backend is region-only so skip_lines is a no-op and accepted via kwargs.
        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {sorted(VARIANTS)}"
            )
        default_repo, default_filename, default_imgsz = VARIANTS[variant]
        model_repo = model_repo or default_repo
        model_filename = model_filename or default_filename
        imgsz = imgsz or default_imgsz

        try:
            from doclayout_yolo import YOLOv10
        except ImportError:
            raise ImportError(
                "doclayout-yolo is not installed. Install with:\n"
                '  pip install "newspaper-ocr[doclayout]"\n'
                "  or: pip install doclayout-yolo"
            )
        from huggingface_hub import hf_hub_download

        self.imgsz = imgsz
        self.conf = conf
        if device is None:
            try:
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
        self.model = _load_yolov10(YOLOv10, model_path)

    def detect(self, image: Image.Image) -> PageLayout:
        w, h = image.size

        # Pass the PIL image, not a numpy array: doclayout-yolo/ultralytics reads
        # ndarrays as BGR (the cv2 convention) but PIL images as RGB, so handing
        # it an RGB ndarray would silently swap the R and B channels.
        results = self.model.predict(
            image.convert("RGB"),
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        regions: list[Region] = []
        for result in results:
            names = getattr(result, "names", {})
            for box in result.boxes:
                x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 <= x0 or y1 <= y0:
                    continue

                cls_id = int(box.cls[0])
                label = str(names.get(cls_id, cls_id)).replace(" ", "_")
                crop = image.crop((x0, y0, x1, y1))
                regions.append(
                    Region(
                        bbox=BBox(x0, y0, x1, y1),
                        image=crop,
                        label=label,
                        lines=[],
                        confidence=float(box.conf[0]),
                    )
                )

        # Region-only detector: no lines, so leave lines_detected False and let
        # every region through to region-level OCR.
        return PageLayout(
            image=image, regions=regions, width=w, height=h, lines_detected=False
        )
