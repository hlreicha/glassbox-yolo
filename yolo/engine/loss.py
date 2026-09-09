import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np

from yolo.data.dataset_utils import cxcywh_to_xyxy
from yolo.engine.utils import bbox_iou, feats_to_boxes,bboxes_to_feats, generate_anchors, TaskAlignedAssigner


class DFLoss(nn.Module): 
    def __init__(self,
                 reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
    
    def __call__(self,
                 pred_dist : torch.Tensor,
                 targets : torch.Tensor):
        '''
        preds: (N * 4,(reg_max))
        targets: (N,4)
        Return sum of left and right DFL losses.
        '''
        target_clamped = targets.clamp(0, self.reg_max - 1 - 0.01)
        target_left = target_clamped.long()
        target_right = target_left + 1
        weight_left = target_right - target_clamped
        weight_right = 1 - weight_left

        loss = ((F.cross_entropy(pred_dist, target_left.view(-1), reduction="none").view(target_left.shape) * weight_left)
                + (F.cross_entropy(pred_dist, target_right.view(-1), reduction="none").view(target_left.shape) * weight_right))
        return loss.mean(-1, keepdim=True)
    
class BoundingBoxLoss(nn.Module):
    def __init__(self,
                 reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.dfl = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self,
                preds_dist: torch.Tensor,
                preds_bboxes: torch.Tensor,
                anchors: torch.Tensor,
                bbox_targets: torch.Tensor,
                score_targets: torch.Tensor,
                sum_target_scores: torch.Tensor,
                mask_fg: torch.Tensor):
        
        weight = score_targets.sum(-1)[mask_fg].unsqueeze(-1)
        iou = bbox_iou(preds_bboxes[mask_fg], bbox_targets[mask_fg], do_CIou=True)
        iou_loss = ((1 - iou) * weight).sum() / sum_target_scores

        target_dist = bboxes_to_feats(bbox_targets, anchors, self.reg_max - 1)
        dfl_loss = self.dfl(preds_dist[mask_fg].view(-1,self.reg_max), target_dist[mask_fg]) * weight
        dfl_loss = dfl_loss.sum() / sum_target_scores

        return iou_loss, dfl_loss
    
    

class DetectionLoss:
    def __init__(self,
                 model,
                 img_size: tuple = (640, 640),
                 number_of_classes: int = 80,
                 reg_max: int = 16,
                 strides: torch.Tensor = torch.tensor([8,16,32]),
                 box_gain: float = 0.05,
                 cls_gain: float = 0.5,
                 dfl_gain: float = 1.0,
                 topk: int = 10,
                 expand_small_boxes: bool = False):
        
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.device = next(model.parameters()).device
        self.img_size = torch.tensor(img_size, dtype=torch.float, device=self.device)
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.topk = topk
        self.strides = strides
        self.num_classes = number_of_classes
        self.reg_max = reg_max
        self.no = number_of_classes + reg_max * 4

        self.do_dfl = reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=self.topk,
                                            number_of_classes=self.num_classes,
                                            expand_small_boxes=expand_small_boxes,
                                            alpha = 0.5) 
        self.bbox_loss = BoundingBoxLoss(reg_max).to(self.device)
        self.proj = torch.arange(reg_max, dtype=torch.float, device=self.device)


    def preprocess(self, ground_truths, batch, scale):
        num_gts, len_of_gt = ground_truths.shape
        if num_gts == 0:
            return torch.zeros(batch, 0, len_of_gt - 1, device= self.device)
        else:
            img_ids = ground_truths[:, 0].to(torch.int32)
            _, unique_counts = img_ids.unique(return_counts=True)
            result = torch.zeros(batch, unique_counts.max(), len_of_gt - 1, device=self.device)

            for i in range(batch):
                mask = (img_ids == i).nonzero(as_tuple=True)[0]
                count = mask.numel() 
                if count:
                    result[i, :count] = ground_truths[mask, 1:]
            result[..., 1:5] = cxcywh_to_xyxy(result[..., 1:5] * scale)
            # result shape is: (batch, max_num_gts, 5) [class, x1, y1, x2, y2]
            return result
        
    def decode_bbox(self,
                    anchor_points: torch.Tensor,
                    pred_dist: torch.Tensor):
        if self.do_dfl:
            batch, anchors, channels = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch, anchors, 4, channels // 4), dim=-1) @ self.proj.type(pred_dist.dtype)
        
        output = feats_to_boxes(pred_dist, anchor_points, xywh=False)
        return output
    

    def __call__(self,
                 preds: tuple,
                 targets):
        # preds: (B, reg_max*4 + num_classes, H, W)
        # targets: (N, 6) [batch_index, class, x, y, w, h]
        batch = preds[0].shape[0]
        dtype = preds[0].dtype
        anchor_points, stride_tensor = generate_anchors(preds, self.strides, 0.5)
        
        preds = torch.cat([x.view(batch, self.no, -1) for x in preds], 2)
        dists,scores = preds.split((self.reg_max * 4, self.num_classes), 1)
        dists,scores = dists.transpose(1,2).contiguous(), scores.transpose(1,2).contiguous()
        pred_boxes = self.decode_bbox(anchor_points, dists) 

        # (batch, max_num_gts, 5) [class, x1, y1, x2, y2]
        targets = self.preprocess(targets, batch, scale = self.img_size[[1, 0, 1, 0]])
        # (batch, max_num_gts, 1), (batch,max_num_gts, 4)
        target_labels, target_boxes = targets.split((1,4),2) 
        mask_fg = torch.where(target_boxes.sum(2, keepdim = True) > 0, True, False)

        #return label_targets,bbox_targets,score_targets,mask_fg.bool(), gt_idx
        # task aligned assigner
        _,bbox_targets, score_targets, mask_fg, gt_idx = self.assigner(
            pred_scores = scores.detach().sigmoid(), 
            pred_bboxes = (pred_boxes.detach() * stride_tensor).type(target_boxes.dtype), 
            anchor_points = anchor_points * stride_tensor,
            target_labels = target_labels, 
            target_bboxes = target_boxes, 
            gt_mask = mask_fg)
       
       
        sum_target_scores = max(score_targets.sum(), 1.0)

        cls_loss = self.bce(scores,score_targets.to(dtype)).sum() / sum_target_scores

        bbox_targets /= stride_tensor
        box_loss, dfl_loss = (
            self.bbox_loss(
                preds_dist = dists,
                preds_bboxes = pred_boxes,
                anchors=anchor_points,
                bbox_targets = bbox_targets,
                score_targets =score_targets,
                sum_target_scores = sum_target_scores,
                mask_fg = mask_fg
            ) if mask_fg.any()
            else (torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)) #else (torch.tensor(0.0, device=self.device),(torch.tensor(0.0, device=self.device)))
        )

        total = (box_loss * self.box_gain + cls_loss * self.cls_gain + dfl_loss * self.dfl_gain) * batch
        return total, torch.stack([box_loss * self.box_gain, cls_loss * self.cls_gain, dfl_loss * self.dfl_gain]).detach()
    