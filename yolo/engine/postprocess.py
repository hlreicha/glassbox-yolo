from typing import Tuple
import torchvision
import torch
from yolo.data.dataset_utils import cxcywh_to_xyxy, denormalize

def nms(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
    number_of_classes=0, 
    max_nms=30000,
    max_wh=7680,
):
    bs = prediction.shape[0]  
    number_of_classes = number_of_classes or (prediction.shape[1] - 4) 

    # get candidates
    candidates = prediction[:, 4:4 + number_of_classes].amax(1) > conf_thres 

    multi_label &= number_of_classes > 1  

    prediction[:,:4,:] = cxcywh_to_xyxy(prediction[:,:4,:], dim=1) 

    prediction = prediction.transpose(-1, -2)

    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for index, pred in enumerate(prediction):  

        pred = pred[candidates[index]]  

        
        # If none remain process next image
        if not pred.shape[0]:
            continue

        box, cls = pred.split((4,number_of_classes), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            pred = torch.cat((box[i], pred[i, 4 + j, None], j[:, None].float()), 1)
        else:  
            conf, j = cls.max(1, keepdim=True)
            pred = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            pred = pred[(pred[:, 5:6] == classes).any(1)]

        # Check shape
        num_boxes = pred.shape[0]
        
        # If no boxes, skip.
        if not num_boxes:  
            continue
            
        if num_boxes > max_nms:  
            pred = pred[pred[:, 4].argsort(descending=True)[:max_nms]]  

        # Batched NMS
        class_offset = pred[:, 5:6] * (0 if agnostic else max_wh) 
        scores = pred[:, 4] 

        # boxes (offset by class)
        boxes = pred[:, :4] + class_offset  
        nms_indices = torchvision.ops.nms(boxes, scores, iou_thres)  
        
        nms_indices = nms_indices[:max_det]

        output[index] = pred[nms_indices]
    return output

def decode_bboxes(targets: torch.Tensor,
                  img_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Convert collated GT targets from normalized xywh to pixel xyxy.

    Args:
        targets: Tensor [N, 6] = [batch_idx, cls, cx, cy, w, h] with bbox
            columns normalized to [0, 1] against the per-batch padded canvas.
        img_shape: ``(H, W)`` of the canvas (e.g. ``images.shape[2:]``).
            Works for fixed imgsz, letterbox=False, and rect/variable
            per-batch shapes — pass the actual batch tensor's H, W.

    Returns:
        New tensor [N, 6] with bbox columns as pixel xyxy. Input not mutated.
    """
    out = targets.clone()
    if out.numel() == 0:
        return out
    H, W = int(img_shape[0]), int(img_shape[1])

    out[:, 2:] = denormalize(out[:, 2:], W, H)
    out[:, 2:] = cxcywh_to_xyxy(out[:, 2:])
    return out