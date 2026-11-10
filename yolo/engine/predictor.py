from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import torch

from .postprocess import nms

PAD_COLOR = (114, 114, 114)


def letterbox_image(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
    """Resize to fit imgsz x imgsz keeping aspect ratio. Returns (canvas, ratio, left, top)."""
    h, w = image.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    if (new_w, new_h) != (w, h):
        interp = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
    top = int(round((imgsz - new_h) / 2 - 0.1))
    left = int(round((imgsz - new_w) / 2 - 0.1))
    canvas = cv2.copyMakeBorder(
        image, top, imgsz - new_h - top, left, imgsz - new_w - left, cv2.BORDER_CONSTANT, value=PAD_COLOR
    )
    return canvas, ratio, left, top


class Predictor:
    def __init__(self,
                 model: torch.nn.Module,
                 imgsz: int = 640,
                 conf: float = 0.25,
                 iou: float = 0.45,
                 max_det: int = 300) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """Detect objects in a BGR image. Returns (N, 6) rows of x1, y1, x2, y2, conf, cls in original pixels."""
        h, w = image_bgr.shape[:2]
        canvas, ratio, left, top = letterbox_image(image_bgr, self.imgsz)
        rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
        batch = torch.from_numpy(rgb).to(self.device).float().div_(255.0).unsqueeze(0)

        out = self.model(batch)
        decoded = out[0] if isinstance(out, (tuple, list)) else out
        det = nms(decoded.float(), conf_thres=self.conf, iou_thres=self.iou, max_det=self.max_det)[0].cpu().numpy()

        det[:, [0, 2]] = ((det[:, [0, 2]] - left) / ratio).clip(0, w)
        det[:, [1, 3]] = ((det[:, [1, 3]] - top) / ratio).clip(0, h)
        return det


def class_color(cls: int) -> tuple[int, int, int]:
    return (37 * cls + 50) % 200 + 55, (91 * cls + 120) % 200 + 55, (17 * cls + 200) % 200 + 55


def draw_detections(image_bgr: np.ndarray, detections: np.ndarray, names: Dict[int, str]) -> np.ndarray:
    out = image_bgr.copy()
    thickness = max(1, round(sum(out.shape[:2]) / 1000))
    font_scale = thickness / 3 + 0.3
    for x1, y1, x2, y2, conf, cls in detections:
        cls = int(cls)
        color = class_color(cls)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, thickness, cv2.LINE_AA)
        label = f"{names.get(cls, cls)} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(p1[1] - th - baseline, 0)
        cv2.rectangle(out, (p1[0], top), (p1[0] + tw, top + th + baseline), color, -1)
        cv2.putText(out, label, (p1[0], top + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return out
