import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

from ..core.logger import get_logger
from yolo.data.dataset_utils import cxcywh_to_xyxy, xyxy_to_cxcywh

logger = get_logger()


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        setattr(a, k, v)


def bbox_iou(box1, box2, do_CIou=False, eps=1e-7):
    
    box1_x1, box1_y1, box1_x2, box1_y2 = box1.split(1, dim = -1)
    box2_x1, box2_y1, box2_x2, box2_y2 = box2.split(1, dim = -1)
    
    w1 = box1_x2 - box1_x1 
    h1 = box1_y2 - box1_y1 + eps
    w2 = box2_x2 - box2_x1 
    h2 = box2_y2 - box2_y1 + eps
    
    # Find Intersection area
    inter = torch.clamp((torch.min(box1_x2,box2_x2) - torch.max(box1_x1,box2_x1)),0) * torch.clamp((torch.min(box1_y2,box2_y2) - torch.max(box1_y1,box2_y1)), 0)
    
    
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    # IoU a lot of money.
    iou = inter / union
    
    if do_CIou:
        # Find the smallest enclosing box
        enclosing_w = torch.max(box1_x2,box2_x2) - torch.min(box1_x1,box2_x1)
        enclosing_h = torch.max(box1_y2,box2_y2) - torch.min(box1_y1,box2_y1)
    
        # Diagonal squared of the enclosing box
        c2 = enclosing_w ** 2 + enclosing_h ** 2 + eps
    
        # Compute square distances between box centers
        center_distance = ((((box2_x1 + box2_x2) / 2) - ((box1_x1 + box1_x2) / 2)) ** 2) + ((((box2_y1+box2_y2) / 2) - ((box1_y1+box1_y2) / 2)) ** 2)
    
        # Aspect ratio consistency term
        v = (4 / math.pi ** 2) * (torch.atan(w2/h2) - torch.atan(w1/h1)) ** 2
    
        # Penalizes the aspect ratio term
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        # CIoU calculation
        return iou - (center_distance / c2 + v * alpha)  
    return iou

def feats_to_boxes(feats_distance:torch.tensor,
                   anchors:torch.tensor,
                   xywh: bool=True, 
                   dim: float = -1) -> torch.tensor:
    dist_lt,dist_rb = feats_distance.split(2,dim)
    x1y1 = anchors - dist_lt
    x2y2 = anchors + dist_rb
    if xywh:
        center = (x1y1 + x2y2) / 2
        width_height = x2y2 - x1y1
        # xywh bbox
        return torch.cat((center, width_height), dim) 
    # xyxy bbox 
    return torch.cat((x1y1, x2y2), dim)  

def bboxes_to_feats(bbox: torch.tensor,
                    anchors: torch.tensor,
                    reg_max: int = 16):
    x1y1, xy2y2 = bbox.split(2, dim=-1)
    return torch.clamp(torch.cat((anchors - x1y1, xy2y2 - anchors), dim=-1), min=0, max=reg_max - 0.01)

def generate_anchors(xs: torch.tensor, 
                     strides: torch.tensor, 
                     offset: float = 0.5):
    anchor_list, stride_tensor = [], []

    dtype, device = xs[0].dtype, xs[0].device

    for index, stride in enumerate(strides):
        h,w = xs[index].shape[2:]
        x = torch.tensor([i + offset for i in range(w)],dtype=dtype).to(device)
        y = torch.tensor([i + offset for i in range(h)],dtype=dtype).to(device)
        sy_b = y[:,None]
        sx_b = x[None,:]
        anchor_list.append(torch.stack((sx_b.expand(h, w), sy_b.expand(h, w)), dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_list), torch.cat(stride_tensor)


class TaskAlignedAssigner(nn.Module):
    def __init__(self,
                 topk: int = 10,
                 number_of_classes: int = 80,
                 expand_small_boxes: bool = False,
                 alpha: float =1.0,
                 beta: float = 6.0,
                 eps: float = 1e-9):
        super().__init__()
        topk2 = None
        self.topk = topk
        self.topk2 = topk2 or topk
        self.number_of_classes = number_of_classes
        self.expand_small_boxes = expand_small_boxes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.strides = torch.tensor([8,16,32])
        #self.stride_val = self.strides[0].item()

    def get_candidates_in_targets(self, 
                                  target_bboxes: torch.Tensor,
                                  anchor_points: torch.Tensor,
                                  gt_mask,
                                  expand_small_boxes: bool = False):
        '''
        For each target bbox, find which anchor points are inside the bbox.
        targetr_bboxes: (batch_size, max_num_targets, 4)
        anchor_points: (num_anchors, 2)
        '''
        if expand_small_boxes:
            target_shape = target_bboxes.shape
            target_bboxes_xywh = xyxy_to_cxcywh(target_bboxes.reshape(-1, 4).clone()).view(target_shape) 
            mask_wh = ((target_bboxes_xywh[..., 2:] < self.strides[0]) * gt_mask).bool()
            target_bboxes_xywh[..., 2:].masked_fill_(mask_wh, self.strides[1].to(dtype=target_bboxes_xywh.dtype, device=target_bboxes_xywh.device))
            target_bboxes = cxcywh_to_xyxy(target_bboxes_xywh.reshape(-1, 4)).view(target_shape)

        x1 = target_bboxes[:,:,0].unsqueeze(-1) < anchor_points[:,0]
        y1 = target_bboxes[:,:,1].unsqueeze(-1) < anchor_points[:,1]
        x2 = target_bboxes[:,:,2].unsqueeze(-1) > anchor_points[:,0]
        y2 = target_bboxes[:,:,3].unsqueeze(-1) > anchor_points[:,1]
        result = (x1 & y1) & (x2 & y2)
        return result.to(torch.float32)
    
    def compute_alignment_metric(self,
                                 pred_scores: torch.Tensor,
                                 pred_bboxes: torch.Tensor,
                                 target_labels: torch.Tensor,
                                 target_bboxes: torch.Tensor,
                                 gt_mask: torch.Tensor):
        '''
        Compute the alignment metric between predicted boxes and target boxes.
        pred_scores: (batch_size, num_anchors, num_classes)
        pred_bboxes: (batch_size, num_anchors, 4)
        target_labels: (batch_size, max_num_targets, 1)
        target_bboxes: (batch_size, max_num_targets, 4)
        gt_mask: (batch_size, max_num_targets, num_anchors)
        '''
        num_anchors = pred_scores.shape[1]
        gt_mask = gt_mask.bool()
        overlap_bboxes = torch.zeros((self.batch_size, self.max_num_targets, num_anchors), dtype=pred_bboxes.dtype, device=pred_bboxes.device)
        target_labels_onehot = F.one_hot(target_labels.squeeze(-1).long(), num_classes=self.number_of_classes).float()
        bbox_scores = torch.einsum('bac,bmc -> bma', pred_scores, target_labels_onehot)
        bbox_scores.masked_fill_(~gt_mask, 0)
        b_idx, m_idx, a_idx = torch.where(gt_mask)
        aligned_pred_bboxes = pred_bboxes[b_idx, a_idx]
        aligned_target_bboxes = target_bboxes[b_idx, m_idx]
        overlap_bboxes[gt_mask] = torch.clamp(bbox_iou(aligned_target_bboxes,aligned_pred_bboxes, do_CIou=True).squeeze(-1),0)

        # alignment metric
        align_metric = (bbox_scores ** self.alpha) * (overlap_bboxes ** self.beta)
        return align_metric, overlap_bboxes
        

    def select_topk(self,
                      align_metric: torch.Tensor,
                      do_largest: bool = True,
                      top_k_mask: torch.Tensor = None):
        metrics, idxs = torch.topk(align_metric, self.topk, dim=2, largest=do_largest)

        top_k_mask = top_k_mask.bool()
        idxs = idxs * top_k_mask
        b_idx, m_idx, k_idx = torch.where(top_k_mask)
        anchor_idx = idxs[b_idx,m_idx,k_idx]
        count = torch.zeros(align_metric.shape, dtype=torch.int8, device=idxs.device)
        count[b_idx, m_idx, anchor_idx] = 1
        duplicates =(~top_k_mask).sum(dim=-1).to(count.dtype)
        count[:,:,0] += duplicates
        count.masked_fill_(count > 1, 0)
        return count.to(align_metric.dtype) 
    
        # Works but computationally expensive.
        # count = F.one_hot(idxs,num_classes=metrics.shape[-1]).sum(-2)


    def get_pos_mask(self,
                     pred_scores: torch.Tensor,
                     pred_bboxes: torch.Tensor,
                     target_labels: torch.Tensor,
                     target_bboxes: torch.Tensor,
                     anchor_points: torch.Tensor,
                     gt_mask: torch.Tensor):
        in_gt_mask = self.get_candidates_in_targets(target_bboxes, anchor_points, gt_mask=gt_mask, expand_small_boxes=self.expand_small_boxes,)

        align_metric, overlaps = self.compute_alignment_metric(pred_scores,
                                                                pred_bboxes,
                                                                target_labels,
                                                                target_bboxes,
                                                                gt_mask * in_gt_mask)

        topk = self.select_topk(align_metric, top_k_mask = gt_mask.expand(-1, -1, self.topk))

        mask_position = topk * in_gt_mask * gt_mask
        return mask_position, align_metric, overlaps
    
    def get_highest_overlaps(self, 
                             pos_mask: torch.Tensor, 
                             overlaps: torch.Tensor, 
                             max_num_targets: int,
                             align_metric: torch.Tensor):
        '''
        Get anchor boxes with highest IuO when assigned to multiple ground truth boxes
        pos_mask: (batch_size, max_num_targets, num_anchors)
        overlaps: (batch_size, max_num_targets, num_anchors)
        max_num_targets: int
        aign_metric: (batch_size, max_num_targets, num_anchors)
        '''
        foreground_mask = pos_mask.sum(-2)
        if foreground_mask.max() > 1:
            topk_overlaps_indices = overlaps.argmax(1).unsqueeze(1)
            mask_scatter = torch.zeros_like(pos_mask).scatter_(1, topk_overlaps_indices, 1)

            
            mask_multi_gts = torch.where(foreground_mask.unsqueeze(1) > 1, True, False).expand(-1, max_num_targets, -1)
            pos_mask = torch.where(mask_multi_gts, mask_scatter, pos_mask)
            foreground_mask = pos_mask.sum(-2)
            #return new_pos_mask.argmax(-2),foreground_mask, new_pos_mask
        if self.topk2 != self.topk:
            align_metric = align_metric * pos_mask
            _,max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True)
            topk2_idxs = torch.zeros_like(pos_mask).scatter_(-1, max_overlaps_idx, 1.0)
            pos_mask *= topk2_idxs
            foreground_mask = pos_mask.sum(-2)

        topk_overlaps_indices = pos_mask.argmax(-2)

        return topk_overlaps_indices, foreground_mask, pos_mask


        
    def get_targets(self,
                    target_labels: torch.Tensor,
                    target_bboxes: torch.Tensor,
                    gt_idx: torch.Tensor,
                    mask_fg: torch.Tensor):
        '''
        target_labels: (batch_size, max_num_targets, 1)
        target_bboxes: (batch_size, max_num_targets, 4)
        gt_idx: (batch_size, num_anchors)
        mask_fg: (batch_size, num_anchors)
        '''
        target_labels = torch.gather(target_labels.long(), 1, gt_idx.unsqueeze(-1)).flatten(-2)
        target_labels = torch.clamp(target_labels, min=0, max=self.number_of_classes)

        target_bboxes = torch.gather(target_bboxes,1,gt_idx.unsqueeze(-1).expand(-1,-1,target_bboxes.shape[-1]))
        target_scores = torch.zeros(self.batch_size, target_labels.shape[1], self.number_of_classes,
                                     dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        target_scores = target_scores * mask_fg.unsqueeze(-1)

        return target_labels, target_bboxes, target_scores

    def normalize_scores(self, 
                         score_targets: torch.Tensor, 
                         align_metric: torch.Tensor,
                         gt_idx: torch.Tensor,
                         mask_fg: torch.Tensor, 
                         pos_mask: torch.Tensor,
                         overlaps: torch.Tensor):
        
        align_metric *= pos_mask
        align_metrics_pos = align_metric.amax(dim=-1, keepdim = True)
        overlaps_pos = (overlaps * pos_mask).amax(dim=-1, keepdim = True)
        align_values = torch.gather(align_metric,1,gt_idx.unsqueeze(1)).squeeze(1)
        align_gt = torch.gather(align_metrics_pos.squeeze(-1),1,gt_idx)
        align_overlaps = torch.gather(overlaps_pos.squeeze(-1),1,gt_idx)
        norm = (align_values * align_overlaps / (align_gt + self.eps)) * mask_fg
        score_targets.mul_(norm.unsqueeze(-1))
        return score_targets
    
    @torch.no_grad()
    def forward(self,
                pred_scores: torch.Tensor,
                pred_bboxes: torch.Tensor,
                anchor_points: torch.Tensor,
                target_labels: torch.Tensor,
                target_bboxes: torch.Tensor,
                gt_mask: torch.Tensor,
                ) -> tuple:
        '''
        pred_scores: (batch_size, num_anchors, num_classes)
        pred_bboxes: (batch_size, num_anchors, 4)
        anchor_points: (num_anchors, 2)
        target_labels: (batch_size, max_num_targets, 1)
        target_bboxes: (batch_size, max_num_targets, 4)
        gt_mask: (batch_size, max_num_targets, 1)
        '''
        self.batch_size = pred_scores.shape[0]
        self.max_num_targets = target_labels.shape[1]
        device = pred_scores.device

        if self.max_num_targets == 0:
            return (torch.full_like(pred_scores[:,:,0], self.number_of_classes),
                    torch.zeros_like(pred_bboxes),
                    torch.zeros_like(pred_scores),
                    torch.zeros_like(pred_scores[:,:,0]),
                    torch.zeros_like(pred_scores[:,:,0]))
        
        pos_mask,align_metric,overlaps = self.get_pos_mask(pred_scores,
                                                           pred_bboxes,
                                                           target_labels,
                                                           target_bboxes,
                                                           anchor_points,
                                                           gt_mask)
        
        gt_idx, mask_fg, pos_mask = self.get_highest_overlaps(pos_mask,
                                                              overlaps,
                                                              self.max_num_targets,
                                                              align_metric)
        

        label_targets, bbox_targets, score_targets = self.get_targets(target_labels, 
                                                                      target_bboxes, 
                                                                      gt_idx, 
                                                                      mask_fg)
        
        score_targets = self.normalize_scores(score_targets, 
                                              align_metric, 
                                              gt_idx, 
                                              mask_fg, 
                                              pos_mask, 
                                              overlaps)

        return label_targets,bbox_targets,score_targets,mask_fg.bool(), gt_idx
    

class EMAModel:
    def __init__(self,
                 model,
                 do_ema: bool = True,
                 alpha: float = 0.9999,
                 tau: float = 2000,
                 update_after_step: int = 0):
        self.ema_model = deepcopy(model).eval()
        self.update_after_step = update_after_step
        self.tau = tau
        self.do_ema = do_ema
        self.alpha = alpha

        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def calculate_decay(self):
        return self.alpha * (1 - math.exp(-self.update_after_step / self.tau))
    
    def update(self, model):
        if not self.do_ema:
            return
        self.update_after_step += 1
        decay = self.calculate_decay()

        ema_sd = self.ema_model.state_dict()
        model_sd = model.state_dict()
        if not hasattr(self, "_fp_keys"):
            self._fp_keys = [k for k, v in ema_sd.items() if v.is_floating_point()]

        ema_tensors = [ema_sd[k] for k in self._fp_keys]
        model_tensors = [model_sd[k].detach() for k in self._fp_keys]

        torch._foreach_mul_(ema_tensors, decay)
        torch._foreach_add_(ema_tensors, model_tensors, alpha=1 - decay)

    def update_attribute(self, model, include=(), exclude=()):
        if self.do_ema:
            copy_attr(self.ema_model, model, include, exclude)
        

class EarlyStopping:
    def __init__(self,
                 patience:int = 20):
        
        self.best_score = 0.0
        self.patience = patience
        self.counter = 0
        self.best_epoch = 0
        self.potential_stop = False
    def __call__(self, 
                 score: float,
                 epoch: int):
        stop_early = False
        if score > self.best_score:
            self.counter = 0
            self.best_epoch = epoch
            self.best_score = score
        else:
            self.counter += 1
            if self.counter >= self.patience:
                stop_early = True
            if self.counter == self.patience - 1:
                self.potential_stop = True
                logger.info(f"Early stopping will be triggered at epoch {epoch + 1} if no improvement is seen. Best score: {self.best_score} at epoch {self.best_epoch}")
        if stop_early:
            logger.info(f"Early stopping at epoch {epoch}. Best score: {self.best_score} at epoch {self.best_epoch}")
        return stop_early





        