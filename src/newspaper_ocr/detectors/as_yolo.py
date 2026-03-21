"""American Stories YOLO-based layout and line detector.

Uses YOLOv8 ONNX models from the American Stories pipeline to detect
layout regions and text lines in newspaper page images.
"""

from __future__ import annotations

import logging
from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.ops import nms

from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.models import BBox, Line, PageLayout, Region

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYOUT_CLASSES: dict[int, str] = {
    0: "article",
    1: "author",
    2: "cartoon_or_advertisement",
    3: "headline",
    4: "image_caption",
    5: "masthead",
    6: "newspaper_header",
    7: "page_number",
    8: "photograph",
    9: "table",
}

LAYOUT_TYPES_FOR_LINES: set[str] = {
    "article",
    "headline",
    "author",
    "image_caption",
}

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "newspaper-ocr" / "models"
_DROPBOX_FALLBACK = Path(
    "/Users/nealcaren/Dropbox/american-stories/american_stories_models"
)

LAYOUT_MODEL_FILENAME = "layout_model_new.onnx"
LINE_MODEL_FILENAME = "line_model_new.onnx"


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------


def _get_onnx_input_name(model_path: str | Path) -> str:
    """Return the input tensor name for an ONNX model."""
    model = onnx.load(str(model_path))
    input_all = [node.name for node in model.graph.input]
    input_init = [node.name for node in model.graph.initializer]
    net_input = list(set(input_all) - set(input_init))
    del model
    return net_input[0]


def _letterbox(
    im: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # (height, width)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, (r, r), (dw, dh)


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from [x_center, y_center, w, h] to [x0, y0, x1, y1]."""
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _nms_yolov8(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    agnostic: bool = False,
) -> list[torch.Tensor]:
    """YOLOv8-format NMS: prediction shape (batch, num_classes+4, num_boxes)."""
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = prediction.shape[1] - 4
    xc = prediction[:, 4 : 4 + nc].amax(1) > conf_thres

    max_wh = 7680
    max_nms = 30000
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x.transpose(0, -1)[xc[xi]]
        if not x.shape[0]:
            continue

        box, cls = x[:, :4], x[:, 4 : 4 + nc]
        box = _xywh2xyxy(box)
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]

    return output


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


def _run_layout_detection(
    session: ort.InferenceSession,
    input_name: str,
    ca_img: np.ndarray,
) -> list[tuple[str, tuple[int, int, int, int], Image.Image]]:
    """Detect layout regions on a full page image (BGR numpy).

    Returns list of (class_label, (x0, y0, x1, y1), pil_crop).
    """
    im = _letterbox(ca_img, (1280, 1280))[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC->CHW, BGR->RGB
    im = np.expand_dims(np.ascontiguousarray(im), axis=0).astype(np.float32) / 255.0

    predictions = session.run(None, {input_name: im})
    predictions = torch.from_numpy(predictions[0])
    predictions = _nms_yolov8(
        predictions, conf_thres=0.01, iou_thres=0.1, max_det=2000, agnostic=True
    )[0]

    bboxes = predictions[:, :4]
    labels = predictions[:, -1]

    layout_img = Image.fromarray(cv2.cvtColor(ca_img, cv2.COLOR_BGR2RGB))
    im_width, im_height = layout_img.size

    # Compute rescaling from 1280x1280 letterboxed coords back to original
    if im_width > im_height:
        w_ratio = 1280.0
        h_ratio = (im_width / im_height) * 1280.0
        w_trans = 0.0
        h_trans = 1280.0 * ((1 - (im_height / im_width)) / 2)
    else:
        h_trans = 0.0
        h_ratio = 1280.0
        w_trans = 1280.0 * ((1 - (im_width / im_height)) / 2)
        w_ratio = 1280.0 * (im_width / im_height)

    results: list[tuple[str, tuple[int, int, int, int], Image.Image]] = []
    for bbox, pred_class in zip(bboxes, labels):
        x0, y0, x1, y1 = torch.round(bbox)
        x0 = int(floor((x0.item() - w_trans) * im_width / w_ratio))
        y0 = int(floor((y0.item() - h_trans) * im_height / h_ratio))
        x1 = int(ceil((x1.item() - w_trans) * im_width / w_ratio))
        y1 = int(ceil((y1.item() - h_trans) * im_height / h_ratio))

        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im_width, x1), min(im_height, y1)

        if x1 <= x0 or y1 <= y0:
            continue

        crop = layout_img.crop((x0, y0, x1, y1))
        class_label = LAYOUT_CLASSES.get(int(pred_class.item()), "unknown")
        results.append((class_label, (x0, y0, x1, y1), crop))

    return results


# ---------------------------------------------------------------------------
# Line detection
# ---------------------------------------------------------------------------


def _get_crops_from_layout_image(
    image: Image.Image,
) -> list[Image.Image]:
    """Chunk tall layout regions into overlapping crops for line detection."""
    im_width, im_height = image.size
    if im_height <= im_width * 2:
        return [image]

    y0 = 0
    y1 = im_width * 2
    crops: list[Image.Image] = []
    while y1 <= im_height:
        crops.append(image.crop((0, y0, im_width, y1)))
        y0 += int(im_width * 1.5)
        y1 += int(im_width * 1.5)
    crops.append(image.crop((0, y0, im_width, im_height)))
    return crops


def _readjust_line_detections(
    line_preds: list[tuple[list[tuple[int, int, int, int]], torch.Tensor, torch.Tensor]],
    orig_img_width: int,
) -> list[tuple[float, float, float, float]]:
    """Merge line detections from overlapping crops back to layout coords."""
    y0 = 0
    dif = int(orig_img_width * 1.5)
    all_preds: list[tuple[float, ...]] = []

    for preds, probs, _labels in line_preds:
        for i, pred in enumerate(preds):
            all_preds.append(
                (pred[0], pred[1] + y0, pred[2], pred[3] + y0, probs[i].item())
            )
        y0 += dif

    all_preds_t = torch.tensor(all_preds)
    final: list[tuple[float, float, float, float]] = []
    if all_preds_t.dim() > 1 and all_preds_t.shape[0] > 0:
        keep = nms(all_preds_t[:, :4], all_preds_t[:, -1], iou_threshold=0.15)
        filtered = all_preds_t[keep, :4]
        filtered = filtered[filtered[:, 1].sort()[1]]  # sort by y
        for pred in filtered:
            px0, py0, px1, py1 = torch.round(pred)
            final.append((px0.item(), py0.item(), px1.item(), py1.item()))

    return final


def _run_line_detection(
    session: ort.InferenceSession,
    input_name: str,
    layout_crops: list[tuple[str, tuple[int, int, int, int], Image.Image]],
) -> list[tuple[int, str, tuple[int, int, int, int], tuple[int, int, int, int], Image.Image]]:
    """Run line detection on text-bearing layout regions.

    Returns list of (layout_idx, class_label, layout_bbox, page_bbox, pil_line_crop).
    """
    all_lines: list[tuple[int, str, tuple[int, int, int, int], tuple[int, int, int, int], Image.Image]] = []

    for layout_idx, (class_label, layout_bbox, layout_crop) in enumerate(layout_crops):
        if class_label not in LAYOUT_TYPES_FOR_LINES:
            continue

        im_width, im_height = layout_crop.size
        chunks = _get_crops_from_layout_image(layout_crop)

        chunk_preds: list[tuple[list[tuple[int, int, int, int]], torch.Tensor, torch.Tensor]] = []
        for chunk in chunks:
            chunk_cv = cv2.cvtColor(np.array(chunk), cv2.COLOR_RGB2BGR)
            im = _letterbox(chunk_cv, (640, 640))[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = (
                np.expand_dims(np.ascontiguousarray(im), axis=0).astype(np.float32)
                / 255.0
            )

            preds = session.run(None, {input_name: im})
            preds = torch.from_numpy(preds[0])
            preds = _nms_yolov8(
                preds, conf_thres=0.2, iou_thres=0.15, max_det=200
            )[0]

            preds = preds[preds[:, 1].sort()[1]]  # sort by y
            line_bboxes = preds[:, :4]
            line_confs = preds[:, -2]
            line_labels = preds[:, -1]

            chunk_w, chunk_h = chunk.size
            if chunk_w > chunk_h:
                h_ratio = (chunk_h / chunk_w) * 640
                h_trans = 640 * ((1 - (chunk_h / chunk_w)) / 2)
            else:
                h_trans = 0.0
                h_ratio = 640.0

            line_proj_crops: list[tuple[int, int, int, int]] = []
            for bbox in line_bboxes:
                bx0, by0, bx1, by1 = torch.round(bbox)
                lx0 = 0
                ly0 = int(floor((by0.item() - h_trans) * chunk_h / h_ratio))
                lx1 = chunk_w
                ly1 = int(ceil((by1.item() - h_trans) * chunk_h / h_ratio))
                line_proj_crops.append((lx0, ly0, lx1, ly1))

            chunk_preds.append((line_proj_crops, line_confs, line_labels))

        line_bboxes_in_layout = _readjust_line_detections(chunk_preds, im_width)

        lx0_page, ly0_page = layout_bbox[0], layout_bbox[1]
        for line_bbox in line_bboxes_in_layout:
            x0 = max(0, int(line_bbox[0]))
            y0 = max(0, int(line_bbox[1]))
            x1 = min(im_width, int(line_bbox[2]))
            y1 = min(im_height, int(line_bbox[3]))

            if x1 <= x0 or y1 <= y0:
                continue

            line_crop = layout_crop.crop((x0, y0, x1, y1))
            if line_crop.size[0] == 0 or line_crop.size[1] == 0:
                continue

            page_bbox = (
                lx0_page + x0,
                ly0_page + y0,
                lx0_page + x1,
                ly0_page + y1,
            )

            all_lines.append(
                (layout_idx, class_label, layout_bbox, page_bbox, line_crop)
            )

    return all_lines


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


def _resolve_model_path(
    explicit: str | Path | None,
    filename: str,
    model_dir: Path | None,
) -> Path:
    """Resolve a model path, checking explicit path, model_dir, cache, and fallback."""
    if explicit is not None:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Model file not found: {p}")

    # Check model_dir if given
    if model_dir is not None:
        p = Path(model_dir) / filename
        if p.is_file():
            return p

    # Check default cache
    p = _DEFAULT_CACHE_DIR / filename
    if p.is_file():
        return p

    # Check Dropbox fallback
    p = _DROPBOX_FALLBACK / filename
    if p.is_file():
        return p

    raise FileNotFoundError(
        f"Could not find {filename}. Searched: "
        f"{model_dir or '(none)'}, {_DEFAULT_CACHE_DIR}, {_DROPBOX_FALLBACK}"
    )


class AsYoloDetector(Detector):
    """American Stories YOLO detector for newspaper layout and line detection.

    Uses YOLOv8 ONNX models to first detect layout regions (articles, headlines,
    etc.) then detect individual text lines within text-bearing regions.

    Parameters
    ----------
    model_dir : str or Path, optional
        Directory containing both layout and line ONNX models.
    layout_model : str or Path, optional
        Explicit path to the layout detection ONNX model.
    line_model : str or Path, optional
        Explicit path to the line detection ONNX model.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        layout_model: str | Path | None = None,
        line_model: str | Path | None = None,
    ):
        model_dir_path = Path(model_dir) if model_dir is not None else None

        layout_path = _resolve_model_path(
            layout_model, LAYOUT_MODEL_FILENAME, model_dir_path
        )
        line_path = _resolve_model_path(
            line_model, LINE_MODEL_FILENAME, model_dir_path
        )

        logger.info("Loading layout model from %s", layout_path)
        self._layout_input_name = _get_onnx_input_name(layout_path)
        self._layout_session = ort.InferenceSession(str(layout_path))

        logger.info("Loading line model from %s", line_path)
        self._line_input_name = _get_onnx_input_name(line_path)
        self._line_session = ort.InferenceSession(str(line_path))

    def detect(self, image: Image.Image) -> PageLayout:
        """Detect layout regions and text lines in a newspaper page image.

        Parameters
        ----------
        image : PIL.Image.Image
            Full page image (RGB).

        Returns
        -------
        PageLayout
            Detected regions with nested lines containing cropped images.
        """
        # Convert PIL RGB to OpenCV BGR
        img_array = np.array(image)
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Layout detection
        layout_crops = _run_layout_detection(
            self._layout_session, self._layout_input_name, img_array
        )
        logger.info("Detected %d layout regions", len(layout_crops))

        # Line detection
        line_results = _run_line_detection(
            self._line_session, self._line_input_name, layout_crops
        )
        logger.info("Detected %d lines", len(line_results))

        # Build PageLayout
        # Group lines by layout_idx
        lines_by_region: dict[int, list[tuple[tuple[int, int, int, int], Image.Image]]] = {}
        for layout_idx, _cls, _layout_bbox, page_bbox, line_crop in line_results:
            lines_by_region.setdefault(layout_idx, []).append((page_bbox, line_crop))

        regions: list[Region] = []
        for layout_idx, (class_label, layout_bbox, layout_crop) in enumerate(
            layout_crops
        ):
            lines: list[Line] = []
            for page_bbox, line_crop in lines_by_region.get(layout_idx, []):
                lines.append(
                    Line(
                        bbox=BBox(*page_bbox),
                        image=line_crop,
                    )
                )

            regions.append(
                Region(
                    bbox=BBox(*layout_bbox),
                    image=layout_crop,
                    label=class_label,
                    lines=lines,
                    confidence=1.0,  # Already passed NMS threshold
                )
            )

        return PageLayout(
            image=image,
            regions=regions,
            width=image.width,
            height=image.height,
        )
