import math
import random
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .dataset_utils import (
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
    scale,
    denormalize,
    normalize,
    add_padding,
    format_image
)


class Compose:
    def __init__(self, transforms) -> None:
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self,
                 data):
        for transform in self.transforms:   
            data = transform(data)
        return data 
    
    def append(self,transform):
        self.transforms.append(transform)
    def insert(self,index,transform):
        self.transforms.insert(index,transform)


class Format:
    """
    Format image and bboxes for YOLO training.
    
    This class converts images to tensors and ensures bboxes are in the correct format
    (normalized xywh) for training.
    """
    def __init__(self, bbox_format: str = 'xywh', normalize: bool = True, batch_idx: bool = True, bgr: float = 0.0) -> None:
        """
        Args:
            bbox_format: Target bbox format ('xywh' or 'xyxy')
            normalize: Whether to normalize bboxes to [0,1]
            batch_idx: Whether to add batch index to output
            bgr: Probability of returning BGR instead of RGB
        """
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.batch_idx = batch_idx
        self.bgr = bgr

    def __call__(self, data: Dict) -> Dict:
        """
        Format image and bboxes for training.
        
        Args:
            data: Dictionary with 'image', 'bboxes', 'cls', 'normalized'
            
        Returns:
            Dictionary with formatted 'img', 'bboxes', 'cls', and optionally 'batch_idx'
        """
        img = data["image"]
        h, w = img.shape[:2]
        
        # Convert image to tensor (C, H, W) and normalize to [0, 1]
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        img = torch.from_numpy(img).float() / 255.0  # Normalize image to [0, 1]
        
        # Handle bboxes
        nl = len(data["bboxes"]) if data["bboxes"] is not None and len(data["bboxes"]) > 0 else 0
        
        if nl > 0:
            bboxes = data["bboxes"].copy()
            cls = data["cls"].copy()
            
            # Bboxes are in xyxy format (absolute coords) after Letterbox/Mosaic
            # Convert to target format
            if self.bbox_format == 'xywh':
                # Convert xyxy -> xywh
                bboxes = xyxy_to_cxcywh(bboxes)
            # else: keep as xyxy
            
            # Normalize if needed
            if self.normalize and not data.get("normalized", False):
                bboxes = normalize(bboxes, w, h)
            elif not self.normalize and data.get("normalized", False):
                bboxes = denormalize(bboxes, w, h)
            
            bboxes_tensor = torch.from_numpy(bboxes).float()
            cls_tensor = torch.from_numpy(cls).float()
        else:
            bboxes_tensor = torch.zeros((0, 4))
            cls_tensor = torch.zeros((0, 1))
        
        # Prepare output
        out = {
            "img": img,
            "cls": cls_tensor,
            "bboxes": bboxes_tensor,
        }
        
        if self.batch_idx:
            out["batch_idx"] = torch.zeros(nl)
        
        return out

class Letterbox:
    def __init__(self, 
                 imgsz: Tuple[int, int] = (640, 640), 
                 do_letterbox: bool = True, 
                 rect: bool = False,
                 debug: bool = False) -> None:
        self.imgsz = imgsz
        self.do_letterbox = do_letterbox
        self.rect = rect
        self.debug = debug

    def __call__(self, data: Dict) -> Dict:
        """
        Resize image while preserving aspect ratio and optionally pad to stride.
        """
        img = data["image"]
        img_h, img_w = img.shape[:2]

        if self.rect and self.do_letterbox:
            new_shape = data.pop("rect_shape", self.imgsz)
        else:
            new_shape = self.imgsz
            data.pop("rect_shape", None)

        color = (114, 114, 114)

        mosaic_applied = data.get("mosaic_applied", False) or data.get("mosaic_border") is not None
        apply_letterbox = self.do_letterbox and not mosaic_applied

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        if apply_letterbox:
            shape = (img_h, img_w)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            ratio = (r, r)
            new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

            dw *= 0.5
            dh *= 0.5
            if (img_w, img_h) != tuple(new_unpad):
                img = cv2.resize(img, new_unpad)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            pad = (dw, dh)
        else:

            img = cv2.resize(img, (new_shape[1], new_shape[0]))
            ratio = (new_shape[1] / img_w, new_shape[0] / img_h)  
            pad = (0.0, 0.0)

        data["image"] = img
        data["resized_shape"] = img.shape[:2]
        data["ratio"] = ratio #None if not apply_letterbox else ratio
        data["pad"] = pad
        if mosaic_applied and not apply_letterbox:
            data["mosaic_border"] = None

        if data.get("bboxes") is not None and len(data["bboxes"]) > 0:

            boxes = data["bboxes"].astype(np.float32).copy()
            if data.get("normalized", False):
                boxes = denormalize(boxes, img_w, img_h)
                data["normalized"] = False

            boxes_are_xyxy = mosaic_applied and not apply_letterbox
            if not boxes_are_xyxy:
                boxes = cxcywh_to_xyxy(boxes)

            boxes = scale(boxes, ratio[0], ratio[1])
            if pad[0] > 0 or pad[1] > 0:
                boxes = add_padding(boxes, pad[0], pad[1])

            # Clip boxes to valid image bounds to avoid floating-point overflow beyond width/height
            h_img, w_img = img.shape[:2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, max(w_img - 1e-3, 0.0))
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, max(h_img - 1e-3, 0.0))

            data["bboxes"] = boxes

        return data

class RandomHSV:
    """
    Randomly adjust HSV (Hue, Saturation, Value) using TorchVision.
    Converts to PIL, applies ColorJitter, converts back to numpy.
    """
    def __init__(self, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4):
        """
        Args:
            hgain: Hue gain (fraction) - will be mapped to [-0.5, 0.5]
            sgain: Saturation gain (fraction) - will be mapped to [1-gain, 1+gain]
            vgain: Value/brightness gain (fraction) - will be mapped to [1-gain, 1+gain]
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, data: Dict) -> Dict:
        """Apply random HSV augmentation using TorchVision ColorJitter."""
        img = data["image"]
        
        if self.hgain or self.sgain or self.vgain:
            # Convert BGR numpy to RGB PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Apply ColorJitter with random parameters
            # Hue: TorchVision expects [-0.5, 0.5]
            hue_factor = random.uniform(-self.hgain, self.hgain) if self.hgain > 0 else 0
            # Saturation and Brightness: TorchVision expects [max(0, 1-factor), 1+factor]
            sat_factor = random.uniform(max(0, 1 - self.sgain), 1 + self.sgain) if self.sgain > 0 else 1.0
            bright_factor = random.uniform(max(0, 1 - self.vgain), 1 + self.vgain) if self.vgain > 0 else 1.0
            
            # Apply transforms
            if hue_factor != 0:
                pil_img = TF.adjust_hue(pil_img, hue_factor)
            if sat_factor != 1.0:
                pil_img = TF.adjust_saturation(pil_img, sat_factor)
            if bright_factor != 1.0:
                pil_img = TF.adjust_brightness(pil_img, bright_factor)
            
            # Convert back to BGR numpy
            img_rgb = np.array(pil_img)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        data["image"] = img
        return data


class RandomFlip:
    """
    Randomly flip image horizontally or vertically using TorchVision.
    Updates bounding boxes in-place (expects xyxy format, absolute pixels).
    """

    def __init__(self, p: float = 0.5, direction: str = 'horizontal'):
        assert direction in ['horizontal', 'vertical'], "direction must be 'horizontal' or 'vertical'"
        assert 0.0 <= p <= 1.0, "p must be in [0, 1]"
        self.p = p
        self.direction = direction

    def __call__(self, data: Dict) -> Dict:
        if random.random() >= self.p:
            return data

        img = data["image"]
        h, w = img.shape[:2]

        if self.direction == 'horizontal':
            img = img[:, ::-1]
        else:
            img = img[::-1]
        data["image"] = np.ascontiguousarray(img)

        bboxes = data.get("bboxes")
        if bboxes is None or len(bboxes) == 0:
            return data

        boxes = bboxes.copy().astype(np.float32)
        if data.get("normalized", False):
            boxes = denormalize(xyxy_to_cxcywh(boxes.copy()), w, h)
            boxes = cxcywh_to_xyxy(boxes)
            data["normalized"] = False

        if self.direction == 'horizontal':
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1
        else:
            y1 = boxes[:, 1].copy()
            y2 = boxes[:, 3].copy()
            boxes[:, 1] = h - y2
            boxes[:, 3] = h - y1

        data["bboxes"] = boxes
        return data


def random_horizontal_flip(data, p: float = 0.5) -> Dict:
    if random.random() < p:
        data["image"] = np.fliplr(data["image"]).copy()
        if data["bboxes"].size:
            data["bboxes"][:, 1] = 1.0 - data["bboxes"][:, 1]
    return data

class Mosaic_simplified:

    def __init__(self, dataset, imgsz=640, p=1.0, pre_transform=None, n=4):

        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        self.dataset = dataset
        self.p = p
        self.pre_transform = pre_transform
        self.imgsz = imgsz

    def __call__(self, data):

        # Probability check - skip if random value exceeds p
        if random.uniform(0, 1) > self.p:
            data["mosaic_applied"] = False
            return data

        data["mosaic_applied"] = True

        # Validate inputs
        assert data.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."

        # Get indexes of 3 additional images to mix with current image (4 total for 2x2 grid)
        buffer_indices = list(self.dataset.buffer)
        current_idx = data.get("sample_idx")
        if current_idx is not None:
            buffer_indices = [idx for idx in buffer_indices if idx != current_idx]

        if not buffer_indices:
            all_indices = list(range(self.dataset.ni))
            if current_idx is not None and len(all_indices) > 1:
                all_indices = [idx for idx in all_indices if idx != current_idx]
            buffer_indices = all_indices

        if not buffer_indices:
            data["mosaic_applied"] = False
            return data

        indexes = random.choices(buffer_indices, k=3)

        # Fetch images and labels from dataset
        mix_labels = [self.dataset._get_label_and_image(i) for i in indexes]
        
        # Apply pre-transform to each fetched image if specified
        if self.pre_transform is not None:
            for i, ml in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(ml)
        
        # Prepare primary image
        primary_img = data['image']
        if data.get("bboxes") is None:
            data["bboxes"] = np.zeros((0, 4), dtype=np.float32)

        data["bboxes"] = cxcywh_to_xyxy(data["bboxes"])
        h, w = data.get("resized_shape", primary_img.shape[:2])
        if data.get("normalized", False):
            data["normalized"] = False
            data["bboxes"] = np.clip(data["bboxes"], 0.0, 1.0)
            data["bboxes"] = denormalize(data["bboxes"], primary_img.shape[1], primary_img.shape[0])
        primary_bboxes = data["bboxes"]  
        primary_cls = data['cls']

        mosaic_metadata = [{
            "image": primary_img,
            "bboxes": primary_bboxes,
            "bbox_classes": primary_cls,
            "resized_shape": (h, w)
        }]
        
        for item in mix_labels:
            if item.get("bboxes") is None:
                item["bboxes"] = np.zeros((0, 4), dtype=np.float32)

            item["bboxes"] = cxcywh_to_xyxy(item["bboxes"])
            h, w = item.get("resized_shape", primary_img.shape[:2])
            if item.get("normalized", False):
                item["normalized"] = False
                item["bboxes"] = np.clip(item["bboxes"], 0.0, 1.0)
                item["bboxes"] = denormalize(item["bboxes"], w, h)
            mosaic_dict = {
                'image': item['image'],
                'bboxes': item["bboxes"],
                'bbox_classes': item['cls'],
                "resized_shape": (h, w)
            }
            mosaic_metadata.append(mosaic_dict)
        
        mosaic_img, mosaic_bboxes, mosaic_classes = self.mosaic(mosaic_metadata)

        data['image'] = mosaic_img
        data['bboxes'] = mosaic_bboxes
        data['cls'] = mosaic_classes
        data['resized_shape'] = (self.imgsz * 2, self.imgsz * 2)
        data['mosaic_border'] = (-self.imgsz // 2, -self.imgsz // 2)
        data['normalized'] = False

        return data

    def mosaic(self, data_dict):

        s = self.imgsz
        random_y_center = int(random.uniform((self.imgsz * 2) * 0.25, (self.imgsz * 2) * 0.75))
        random_x_center = int(random.uniform((self.imgsz * 2) * 0.25, (self.imgsz * 2) * 0.75))
        
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_h, mosaic_w = mosaic_img.shape[:2]
        
        mosaic_bboxes = []
        mosaic_classes = []
        
        for i, item in enumerate(data_dict):
            h, w = item["resized_shape"]
            img = item["image"]
            bboxes = item["bboxes"] 
            cls = item["bbox_classes"]  
            
            # Calculate placement coordinates
            if i == 0:  # top left
                x1l, y1l, x2l, y2l = max(random_x_center - w, 0), max(random_y_center - h, 0), random_x_center, random_y_center
                x1s, y1s, x2s, y2s = w - (x2l - x1l), h - (y2l - y1l), w, h # bottom right of source image
            elif i == 1:  # top right
                x1l, y1l, x2l, y2l = random_x_center, max(random_y_center - h, 0), min(random_x_center + w, s * 2), random_y_center
                x1s, y1s, x2s, y2s = 0, h - (y2l - y1l), min(w, x2l - x1l), h # bottom left of source image
            elif i == 2:  # bottom left
                x1l, y1l, x2l, y2l = max(random_x_center - w, 0), random_y_center, random_x_center, min(s * 2, random_y_center + h)
                x1s, y1s, x2s, y2s = w - (x2l - x1l), 0, w, min(y2l - y1l, h) # top right of source image
            elif i == 3:  # bottom right
                x1l, y1l, x2l, y2l = random_x_center, random_y_center, min(random_x_center + w, s * 2), min(s * 2, random_y_center + h)
                x1s, y1s, x2s, y2s = 0, 0, min(w, x2l - x1l), min(y2l - y1l, h) # top left of source image
            
            # Place image slice into mosaic
            mosaic_img[y1l:y2l, x1l:x2l] = img[y1s:y2s, x1s:x2s]
            
            # Calculate padding for bounding boxes
            padw = x1l - x1s
            padh = y1l - y1s
            
            # Adjust bounding boxes for this image
            if len(bboxes) > 0:
                # Add padding to all bboxes
                adjusted_bboxes = bboxes.copy()
                adjusted_bboxes[:, [0, 2]] += padw 
                adjusted_bboxes[:, [1, 3]] += padh 
                mosaic_bboxes.append(adjusted_bboxes)
                mosaic_classes.append(cls)
        
        if mosaic_bboxes:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, axis=0)
            mosaic_classes = np.concatenate(mosaic_classes, axis=0)
            
            # Clip bboxes to mosaic boundaries
            mosaic_bboxes[:, [0, 2]] = mosaic_bboxes[:, [0, 2]].clip(0, mosaic_w)
            mosaic_bboxes[:, [1, 3]] = mosaic_bboxes[:, [1, 3]].clip(0, mosaic_h)
            
            # Remove zero-area boxes
            areas = (mosaic_bboxes[:, 2] - mosaic_bboxes[:, 0]) * (mosaic_bboxes[:, 3] - mosaic_bboxes[:, 1])
            valid = areas > 0
            mosaic_bboxes = mosaic_bboxes[valid]
            mosaic_classes = mosaic_classes[valid]

            mosaic_bboxes = mosaic_bboxes.astype(np.float32)
            mosaic_classes = mosaic_classes.astype(np.float32)
        else:
            mosaic_bboxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_classes = np.zeros((0, 1), dtype=np.float32)
        
        return mosaic_img, mosaic_bboxes, mosaic_classes


class RandomAffine:
    """Apply random affine transformation after mosaic to mimic YOLO behaviour."""

    def __init__(
        self,
        imgsz: int,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border_value: Tuple[int, int, int] = (114, 114, 114),
        only_mosaic: bool = False,
        pre_transform=None,
        min_bbox_area: float = 4.0,
        min_bbox_side: float = 2.0,
    ) -> None:
        self.imgsz = imgsz
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border_value = border_value
        self.only_mosaic = only_mosaic
        self.pre_transform = pre_transform
        self.min_bbox_area = min_bbox_area
        self.min_bbox_side = min_bbox_side

    @staticmethod
    def _xyxy_to_corners(boxes: np.ndarray) -> np.ndarray:
        tl = np.stack([boxes[:, 0], boxes[:, 1], np.ones(len(boxes))], axis=1)
        tr = np.stack([boxes[:, 2], boxes[:, 1], np.ones(len(boxes))], axis=1)
        bl = np.stack([boxes[:, 0], boxes[:, 3], np.ones(len(boxes))], axis=1)
        br = np.stack([boxes[:, 2], boxes[:, 3], np.ones(len(boxes))], axis=1)
        return np.stack([tl, tr, bl, br], axis=1)

    def _build_matrix(self, width: int, height: int, border: Tuple[int, int]) -> np.ndarray:
        cx, cy = width / 2.0, height / 2.0

        c = np.eye(3, dtype=np.float32)
        c[0, 2] = -cx
        c[1, 2] = -cy

        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        theta = math.radians(angle)
        cos_theta = math.cos(theta) * scale
        sin_theta = math.sin(theta) * scale

        r = np.eye(3, dtype=np.float32)
        r[0, 0] = cos_theta
        r[0, 1] = -sin_theta
        r[1, 0] = sin_theta
        r[1, 1] = cos_theta

        shear_x = math.tan(math.radians(random.uniform(-self.shear, self.shear)))
        shear_y = math.tan(math.radians(random.uniform(-self.shear, self.shear)))
        s = np.eye(3, dtype=np.float32)
        s[0, 1] = shear_x
        s[1, 0] = shear_y

        tx = random.uniform(-self.translate, self.translate) * self.imgsz + border[0]
        ty = random.uniform(-self.translate, self.translate) * self.imgsz + border[1]
        t = np.eye(3, dtype=np.float32)
        t[0, 2] = tx
        t[1, 2] = ty

        m = t @ s @ r @ c

        if self.perspective > 0.0:
            p = np.eye(3, dtype=np.float32)
            p[2, 0] = random.uniform(-self.perspective, self.perspective)
            p[2, 1] = random.uniform(-self.perspective, self.perspective)
            m = p @ m

        # bring back to the final canvas centre
        centre = np.eye(3, dtype=np.float32)
        centre[0, 2] = self.imgsz / 2.0
        centre[1, 2] = self.imgsz / 2.0

        return centre @ m

    def _warp_boxes(self, boxes: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        corners = self._xyxy_to_corners(boxes)
        n = corners.shape[0]
        reshaped = corners.reshape(-1, 3)
        projected = reshaped @ matrix.T
        projected_xy = projected[:, :2] / np.clip(projected[:, 2:3], a_min=1e-6, a_max=None)
        projected_xy = projected_xy.reshape(n, 4, 2)

        x_coords = projected_xy[:, :, 0]
        y_coords = projected_xy[:, :, 1]
        new_boxes = np.stack(
            [
                x_coords.min(axis=1),
                y_coords.min(axis=1),
                x_coords.max(axis=1),
                y_coords.max(axis=1),
            ],
            axis=1,
        )

        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, self.imgsz)
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, self.imgsz)

        new_widths = new_boxes[:, 2] - new_boxes[:, 0]
        new_heights = new_boxes[:, 3] - new_boxes[:, 1]
        new_areas = new_widths * new_heights

        # Original box areas for area-ratio filtering
        orig_widths = boxes[:, 2] - boxes[:, 0]
        orig_heights = boxes[:, 3] - boxes[:, 1]
        orig_areas = orig_widths * orig_heights

        # Filter: min side, min area, area ratio (new/old > 0.10), aspect ratio < 100
        eps = 1e-16
        ar = np.maximum(new_widths / (new_heights + eps), new_heights / (new_widths + eps))
        valid = (
            (new_widths > self.min_bbox_side)
            & (new_heights > self.min_bbox_side)
            & (new_areas > self.min_bbox_area)
            & (new_areas / (orig_areas + eps) > 0.10)
            & (ar < 100)
        )
        return new_boxes.astype(np.float32), valid

    def __call__(self, data: Dict) -> Dict:
        if self.only_mosaic and data.get("mosaic_border", None) is None:
            return data

        # When mosaic was not applied, run pre_transform (Letterbox) first
        if self.pre_transform and "mosaic_border" not in data:
            data = self.pre_transform(data)

        img = data["image"]
        h, w = img.shape[:2]
        border = data.get("mosaic_border", (0, 0))

        matrix = self._build_matrix(w, h, border).astype(np.float32)
        warped = cv2.warpPerspective(img, matrix, (self.imgsz, self.imgsz), borderValue=self.border_value)
        data["image"] = warped
        data["resized_shape"] = (self.imgsz, self.imgsz)
        data["ratio"] = (1.0, 1.0)
        data["pad"] = (0.0, 0.0)

        bboxes = data.get("bboxes")
        if bboxes is not None and len(bboxes) > 0:
            boxes = bboxes.astype(np.float32)
            if data.get("normalized", False):
                boxes_xywh = xyxy_to_cxcywh(boxes.copy())
                boxes_xywh = denormalize(boxes_xywh, w, h)
                boxes = cxcywh_to_xyxy(boxes_xywh)
            warped_boxes, valid = self._warp_boxes(boxes, matrix)
            data["bboxes"] = warped_boxes[valid]
            data["cls"] = data["cls"][valid]
        else:
            data["bboxes"] = np.zeros((0, 4), dtype=np.float32)
            data["cls"] = np.zeros((0, 1), dtype=np.float32)

        data["normalized"] = False
        data.setdefault("mosaic_border", border)

        return data
    
class RandomErasing:
    """
    Randomly erases a rectangular region of an image tensor during training.
    Fills the erased region with random pixel values in [0, 1].

    Args:
        p: Probability of applying the transform.
        scale: Range of proportion of the image area to erase, as (min, max).
        ratio: Range of aspect ratio of the erased region, as (min, max).
        max_attempts: Maximum number of attempts to find a valid region.
    """
    def __init__(self,
                 p: float = 0.4,
                 scale: tuple = (0.02, 0.33),
                 ratio: tuple = (0.3, 3.3),
                 max_attempts: int = 10):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts

    def __call__(self, data: Dict) -> Dict:
        if random.uniform(0, 1) >= self.p:
            return data
        img = data["img"]  # (C, H, W) float tensor in [0, 1]
        c, h, w = img.shape
        area = h * w
        for _ in range(self.max_attempts):
            erase_area = area * random.uniform(*self.scale)
            aspect = random.uniform(*self.ratio)
            eh = int(round(math.sqrt(erase_area * aspect)))
            ew = int(round(math.sqrt(erase_area / aspect)))
            if eh >= h or ew >= w:
                continue
            y0 = random.randint(0, h - eh)
            x0 = random.randint(0, w - ew)
            img[:, y0:y0 + eh, x0:x0 + ew] = torch.rand(c, eh, ew, dtype=img.dtype)
            data["img"] = img
            break
        return data


def build_transforms(dataset, imgsz, hyp, rect=False):
    """
    Build transformation pipeline for YOLO-style training.

        Flow:
            • Optional Mosaic (tagged per-sample)
            • Optional RandomAffine (skipped by default)
            • Letterbox when Mosaic is disabled, direct resize when Mosaic is enabled
            • Colour and flip augmentations
            • Final formatting of tensors/labels
    """

    transforms = []

    mosaic_prob = float(max(hyp.get('mosaic', 0.0), 0.0))
    if mosaic_prob > 0.0:
        transforms.append(
            Mosaic_simplified(
                dataset=dataset,
                imgsz=imgsz,
                p=mosaic_prob,
                pre_transform=None,
            )
        )

    if hyp.get('random_affine', True) or mosaic_prob > 0.0:
        transforms.append(
            RandomAffine(
                imgsz=imgsz,
                degrees=hyp.get('degrees', 0.0),
                translate=hyp.get('translate', 0.1),
                scale=hyp.get('scale', 0.5),
                shear=hyp.get('shear', 0.0),
                perspective=hyp.get('perspective', 0.0),
                only_mosaic=False,
                pre_transform=Letterbox(
                    imgsz=(imgsz, imgsz),
                    do_letterbox=hyp.get('do_letterbox', True),
                    rect=hyp.get('rect', False),
                ),
            )
        )

    transforms.append(
        Letterbox(
            imgsz=(imgsz, imgsz),
            do_letterbox=hyp.get('do_letterbox', True),
            rect=hyp.get('rect', False),
        )
    )

    transforms.append(
        RandomHSV(
            hgain=hyp.get('hsv_h', 0.015),
            sgain=hyp.get('hsv_s', 0.7),
            vgain=hyp.get('hsv_v', 0.4),
        )
    )

    transforms.append(RandomFlip(p=hyp.get('flipud', 0.0), direction='vertical'))
    transforms.append(RandomFlip(p=hyp.get('fliplr', 0.5), direction='horizontal'))

    transforms.append(
        Format(
            bbox_format='xywh',
            normalize=True,
            batch_idx=True,
        )
    )

    if hyp.get('erasing', 0.0) > 0.0:
        transforms.append(RandomErasing(p=hyp.get('erasing', 0.4)))

    return Compose(transforms)
