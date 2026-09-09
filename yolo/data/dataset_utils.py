from collections import abc
from itertools import repeat
from numbers import Number
from typing import List
import random

import numpy as np
import torch

def scale(boxes,scale_w, scale_h):
    """Scales boxes according to given width and height scales."""
    boxes[:, 0] *= scale_w
    boxes[:, 1] *= scale_h
    boxes[:, 2] *= scale_w
    boxes[:, 3] *= scale_h

    return boxes


def denormalize(boxes, w, h):
    """Convert normalized xywh to pixel xywh."""
    assert boxes.shape[1] == 4, "boxes should have shape (n, 4)"
    boxes = np.clip(boxes, 0.0, 1.0) if isinstance(boxes, np.ndarray) else boxes.clamp(0.0, 1.0)
    boxes[:, 0] *= w
    boxes[:, 1] *= h
    boxes[:, 2] *= w
    boxes[:, 3] *= h
    return boxes

def normalize(boxes, w, h):
    """Convert pixel xywh to normalized xywh."""
    assert boxes.shape[1] == 4, "boxes should have shape (n, 4)"
    assert (boxes >= 0).all(), "boxes should be non-negative"
    assert (boxes[..., 0] <= w).all() and (boxes[..., 2] <= w).all(), "x coordinates should be less than image width"
    assert (boxes[..., 1] <= h).all() and (boxes[..., 3] <= h).all(), "y coordinates should be less than image height"
    boxes[:, 0] /= w 
    boxes[:, 1] /= h
    boxes[:, 2] /= w
    boxes[:, 3] /= h
    return boxes

def add_padding(boxes, padw, padh):
    """Add padding to boxes."""
    assert boxes.shape[1] == 4, "boxes should have shape (n, 4)"
    assert padw >= 0 and padh >= 0, "padding should be non-negative"

    boxes[:, 0] += padw
    boxes[:, 1] += padh
    boxes[:, 2] += padw
    boxes[:, 3] += padh
    return boxes

# Convert boxes from (x1, y1, x2, y2) to center-based (cx, cy, w, h).
# Works on numpy arrays and torch tensors.
def xyxy_to_cxcywh(boxes, dim: int = -1):
    assert boxes.shape[dim] == 4, "boxes should have size 4 along the coord axis"
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(dim=dim)
        w = x2 - x1
        h = y2 - y1
        return torch.stack((x1 + w / 2, y1 + h / 2, w, h), dim=dim)
    x1, y1, x2, y2 = np.moveaxis(boxes, dim, 0)
    w = x2 - x1
    h = y2 - y1
    return np.stack((x1 + w / 2, y1 + h / 2, w, h), axis=dim)


# Convert boxes from center-based (cx, cy, w, h) to (x1, y1, x2, y2).
# Works on numpy arrays and torch tensors.
def cxcywh_to_xyxy(boxes, dim: int = -1):
    assert boxes.shape[dim] == 4, "boxes should have size 4 along the coord axis"
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes.unbind(dim=dim)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=dim)
    cx, cy, w, h = np.moveaxis(boxes, dim, 0)
    return np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), axis=dim)

def format_image(image: np.ndarray, 
                 bgr: bool = True) -> torch.Tensor:
    """Format image for model input."""
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=-1)
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image[::-1] if random.uniform(0, 1) > bgr else image)
    image = torch.from_numpy(image)

    return image
