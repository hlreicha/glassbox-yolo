import torch
import numpy as np
from typing import List, Tuple

from yolo.engine.utils import bbox_iou

def interp_vectorized(
        recalls: torch.Tensor, 
        precisions: torch.Tensor, 
        x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized interpolation of precision values at specified recall levels.
    
    Args:
        recalls: Tensor of recall values. Shape: [num_points]
        precisions: Tensor of precision values. Shape: [num_points]
        x: Tensor of recall levels to interpolate. Shape: [num_levels]
    Returns:
        Tensor of interpolated precision values at recall levels x. Shape: [num_levels].    
    """
    # Calculate AP using 101-point interpolation
    n_thresh = recalls.shape[1]
    recall_indexes = torch.searchsorted(
        recalls.T.contiguous(),              
        x.unsqueeze(0).expand(n_thresh, 101),
        right=True                           
    ).clamp(max=recalls.shape[0] - 1)
    idx_left = (recall_indexes - 1).clamp(min=0) 
    recall_left  = torch.gather(recalls.T,    1, idx_left)      
    recall_right = torch.gather(recalls.T,    1, recall_indexes)  
    prec_left    = torch.gather(precisions.T, 1, idx_left)      
    prec_right   = torch.gather(precisions.T, 1, recall_indexes)
    denom = recall_right - recall_left
    t = torch.where(denom == 0, torch.ones_like(denom), (x - recall_left) / denom)
    interp_precisions = prec_left + t * (prec_right - prec_left)
    ap = torch.trapezoid(interp_precisions, x.unsqueeze(0).expand(n_thresh, 101), dim = 1)

    return ap

def match_predictions_to_iou_thresholds(
        corrects: torch.Tensor,
        pred_indices: torch.Tensor,
        pred_conf: torch.Tensor,
        iou: torch.Tensor,
        iou_thresholds: torch.Tensor,
        use_scipy: bool) -> torch.Tensor:
    """
    Match predictions to IoU thresholds and update corrects tensor.
    
    Args:
        corrects: Tensor to update with correct matches. Shape: [num_preds, num_thresholds]
        pred_indices: Tensor of prediction indices. Shape: [num_preds]
        iou: Tensor of IoU values between predictions and targets. Shape: [num_preds, num_targets]
        iou_thresholds: Tensor of IoU thresholds. Shape: [num_thresholds]
        use_scipy: Whether to use scipy for optimal matching.
    """
    if use_scipy:
        from scipy import optimize as opt
        iou_np = iou.cpu().numpy()
        for i, threshold in enumerate(iou_thresholds.tolist()):
            mask = iou_np >= threshold
            iou_masked = iou_np * mask
            if iou_masked.any():
                row_ind, col_ind = opt.linear_sum_assignment(iou_masked, maximize=True)
                valid = iou_masked[row_ind, col_ind] > 0
                if valid.any():
                    corrects[pred_indices[row_ind[valid]], i] = True
    else:
        T = iou_thresholds.numel()
        iou = iou[:, :, None].expand(-1, -1, T) * (iou[:, :, None] >= iou_thresholds[None, None, :])

        dudep = iou.argmax(1).unsqueeze(1)                          
        iou_mask = torch.zeros_like(iou)
        iou_mask.scatter_(1, dudep, 1)
        iou = iou * iou_mask

        score_tensor = pred_conf[:, None, None].expand_as(iou)
        iou_score = torch.where(iou > 0, score_tensor, iou)
        dudep = iou_score.argmax(0).unsqueeze(0)                    
        iou_score_mask = torch.zeros_like(iou_score)
        iou_score_mask.scatter_(0, dudep, 1)
        iou_score = iou_score * iou_score_mask
        iou = torch.where(iou_score > 0, iou, torch.zeros_like(iou))

        flatten_iou = torch.gather(iou, 1, iou.argmax(1, keepdim=True)).squeeze(1)   
        correct = flatten_iou >= iou_thresholds
        corrects[pred_indices] = correct
    return corrects

def process_class_map(class_map: List[dict]) -> List[dict]:
    map_50_class_map = []

    for entry in class_map:
        map_50_class_map.append({
            "class": entry["class"],
            "AP": float(entry["AP"][0]),
            "recalls": float(entry["recalls"][0]),
            "precisions": float(entry["precisions"][0])
        })
    map_50_90_class_map = []
    for entry in class_map:
        ap = entry["AP"]
        rec = entry["recalls"]
        prec = entry["precisions"]
        ap = ap.numpy() if hasattr(ap, "numpy") else np.asarray(ap)
        rec = rec.numpy() if hasattr(rec, "numpy") else np.asarray(rec)
        prec = prec.numpy() if hasattr(prec, "numpy") else np.asarray(prec)
        map_50_90_class_map.append({
            "class": entry["class"],
            "AP": float(np.mean(ap)),
            "recalls": float(np.mean(rec)),
            "precisions": float(np.mean(prec))
        })
    return map_50_class_map, map_50_90_class_map

def mean_avg_precision(preds: List[List], 
                      targets: List[List], 
                      iou_thresholds: torch.Tensor, 
                      num_classes: int = 1) -> Tuple[float, float, float, List]:
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        preds: List of predictions [[train_idx, class, prob, x1, y1, x2, y2], ...]
        targets: List of ground truths [[train_idx, class, x1, y1, x2, y2], ...]
        iou_threshold: IoU threshold for positive detection
        num_classes: Number of classes
    
    Returns:
        tuple: (mAP, mean_precision, mean_recall, class_map)
    """

    if len(preds) == 0 or len(targets) ==0:
        print("Warning: Empty predictions or targets provided.")
        return 0.0, 0.0, 0.0, []
    
    if iou_thresholds is None or iou_thresholds.numel() == 0:
        iou_thresholds = torch.linspace(start=0.5, end=0.95, steps=10)
    
    if num_classes <= 0:
        raise ValueError(f"Number of classes must be positive, got {num_classes}")
    
    average_precisions = []
    epsilon = 1e-16
    
    # Initialize tracking variables
    class_map = []
    precisions_list = []
    recalls_list = []
    
    #preds = torch.tensor(preds,dtype=torch.float32)
    #targets = torch.tensor(targets,dtype=torch.float32)
    targets = targets.clone()
    targets[:, 1] = targets[:, 1].long().to(targets.dtype)
    unique_classes = torch.unique(targets[:,1])
    if unique_classes.max() >= num_classes:
        raise ValueError(f"Found class index {unique_classes.max()} in targets, but num_classes is set to {num_classes}")
    targets_idx = targets[:,0].unique()
    
    corrects = torch.zeros((len(preds), len(iou_thresholds)), dtype=torch.bool)

    for t in targets_idx:
        pred_indices = torch.where(preds[:,0] == t)[0]
        pred_boxes = preds[pred_indices,3:].unsqueeze(1)
        pred_classes = preds[pred_indices,1]
        pred_conf = preds[pred_indices,2]
        targ_indices = torch.where(targets[:,0] == t)[0]
        
        target_boxes = targets[targ_indices,2:].unsqueeze(0)
        target_classes = targets[targ_indices,1]
        correct_classes = pred_classes.unsqueeze(1) == target_classes.unsqueeze(0)

        if len(pred_indices) == 0:
            continue
        iou = bbox_iou(pred_boxes,target_boxes,do_CIou=False).squeeze(-1) * correct_classes
        corrects = match_predictions_to_iou_thresholds(corrects, pred_indices, pred_conf, iou, iou_thresholds, use_scipy=False)


    _,indices = torch.sort(preds[:,2],descending = True)
    preds_sorted = preds[indices]
    corrects_sorted = corrects[indices]
    preds_cls_global = preds_sorted[:,1].long()
    conf_sorted = preds_sorted[:,2]

    n_thresh = len(iou_thresholds)
    prep_0s = torch.zeros(n_thresh)
    prep_1s = torch.ones(n_thresh)
    x = torch.linspace(0, 1, 101)

    # store raw P/R/conf per class so we can find the global best conf threshold later
    class_pr = {}

    # Calculate AP for each class
    for c in range(num_classes):
        ground_truths_for_class = [t for t in targets if t[1] == c]

        if len(ground_truths_for_class) == 0:
            continue
            
        class_mask = preds_cls_global == c
        if class_mask.sum() == 0:
            average_precisions.append(np.zeros(n_thresh))
            class_map.append({"class": c, "AP": np.zeros(n_thresh), "recalls": prep_0s, "precisions": prep_0s})
            continue

        # Get TP/FP for this class
        tp = corrects_sorted[class_mask]
        fp = ~(corrects_sorted[class_mask])

        # Calculate cumulative TP and FP
        tp_cumsum = torch.cumsum(tp.float(), dim=0)
        fp_cumsum = torch.cumsum(fp.float(), dim=0)

        total_gt = len(ground_truths_for_class)

        # Calculate precision and recall
        recalls = tp_cumsum / (total_gt + epsilon)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + epsilon)

        class_pr[c] = (precisions, recalls, conf_sorted[class_mask])

        # Add sentinel points for AUC calculation
        recalls = torch.cat((prep_0s.unsqueeze(0),recalls, prep_1s.unsqueeze(0)),dim=0)
        precisions = torch.cat((prep_1s.unsqueeze(0),precisions, prep_0s.unsqueeze(0)),dim=0)

        # Apply precision envelope 
        precisions = torch.flip(torch.cummax(torch.flip(precisions, dims=[0]), dim=0).values, dims=[0])
        
        ap = interp_vectorized(recalls, precisions, x)
        average_precisions.append(ap)
        class_map.append({"class": c, "AP": ap, "recalls": None, "precisions": None})

    if not average_precisions:
        print("No detections or ground truths found.")
        return 0.0, 0.0, 0.0, []

    # find the global confidence threshold at peak mean F1.
    # pain
    if class_pr:
        x_conf = torch.linspace(0, 1, 1000)
        p_curves = []
        r_curves = []

        for c in class_pr:
            prec_c, rec_c, conf_c = class_pr[c]
            conf_neg = -conf_c

            idx_right_raw = torch.searchsorted(conf_neg.contiguous(), -x_conf, right=True)
            lmask  = idx_right_raw == 0
            rmask  = idx_right_raw >= len(conf_c)
            idx_right = idx_right_raw.clamp(1, len(conf_c) - 1)
            idx_left  = idx_right - 1
            denom  = conf_neg[idx_right] - conf_neg[idx_left]
            t = torch.where(denom == 0, torch.ones_like(denom), (-x_conf - conf_neg[idx_left]) / denom).unsqueeze(1)
            p_curve = prec_c[idx_left] + t * (prec_c[idx_right] - prec_c[idx_left])
            r_curve = rec_c[idx_left]  + t * (rec_c[idx_right]  - rec_c[idx_left])
            p_curve[lmask] = 1.0
            r_curve[lmask] = 0.0
            p_curve[rmask] = prec_c[-1]
            r_curve[rmask] = rec_c[-1]
            p_curves.append(p_curve)
            r_curves.append(r_curve)
        # [nc, 1000, n_thresh]
        p_all = torch.stack(p_curves)   
        r_all = torch.stack(r_curves)

        f1 = 2 * p_all[:,:,0] * r_all[:,:,0] / (p_all[:,:,0] + r_all[:,:,0] + epsilon)
        mean_f1 = f1.mean(0)
        nf = round(len(mean_f1) * 0.1 * 2) // 2 + 1
        kernel = torch.ones(1, 1, nf) / nf
        mean_f1_padded = torch.cat([mean_f1[:1].expand(nf // 2), mean_f1, mean_f1[-1:].expand(nf // 2)])
        mean_f1 = torch.nn.functional.conv1d(mean_f1_padded.view(1, 1, -1), kernel).squeeze()
        best_i = int(mean_f1.argmax())

        precisions_list = list(p_all[:, best_i, :])
        recalls_list    = list(r_all[:, best_i, :])

        # write back P/R into class_map
        pr_idx = 0
        for entry in class_map:
            if entry["recalls"] is None:
                entry["precisions"] = precisions_list[pr_idx]
                entry["recalls"]    = recalls_list[pr_idx]
                pr_idx += 1

    # Calculate mean precision and recall
    mean_precision = torch.stack(precisions_list).mean(0).numpy() if precisions_list else np.zeros(n_thresh)
    mean_recall    = torch.stack(recalls_list).mean(0).numpy()    if recalls_list    else np.zeros(n_thresh)

    return np.mean(average_precisions,axis=0), mean_precision, mean_recall, class_map


def mean_avg_precision_metric(preds: List[List],
                             targets: List[List],
                             num_classes: int = 4,
                             iou_range: Tuple[float, float] = (0.50, 0.95),
                             step: float = 0.05) -> Tuple[float, float, float, List]:
    """
    Calculate mAP over a range of IoU thresholds (COCO-style evaluation).
    
    Args:
        preds: List of predictions
        targets: List of ground truths  
        num_classes: Number of classes
        iou_range: Tuple of (min_iou, max_iou)
        step: Step size for IoU thresholds
        
    Returns:
        tuple: (mean_mAP, mean_precision, mean_recall, class_map)
    """
    if len(preds) == 0 or len(targets) == 0:
        empty = {"map": 0.0, "precision": 0.0, "recall": 0.0, "class_map": []}
        return empty, dict(empty), 0.0
        
    map_50_metrics = {
        "map": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "class_map": []
    }
    map_50_95_metrics = {
        "map": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "class_map": []
    }
    iou_thresholds = np.arange(iou_range[0], iou_range[1] + step, step)

    
    map,mean_precision, mean_recall, class_map = mean_avg_precision(preds, targets, torch.tensor(iou_thresholds), num_classes = num_classes)
    map_50_class_map, map_50_90_class_map = process_class_map(class_map)

    map_50_metrics["map"] = float(map[0]) if len(map) > 0 else 0.0
    map_50_metrics["precision"] = float(mean_precision[0]) if len(mean_precision) > 0 else 0.0
    map_50_metrics["recall"] = float(mean_recall[0]) if len(mean_recall) > 0 else 0.0
    map_50_metrics["class_map"] = map_50_class_map
    map_50_95_metrics["map"] = float(np.mean(map)) if len(map) > 0 else 0.0
    map_50_95_metrics["precision"] = float(np.mean(mean_precision)) if len(mean_precision) > 0 else 0.0
    map_50_95_metrics["recall"] = float(np.mean(mean_recall)) if len(mean_recall) > 0 else 0.0
    map_50_95_metrics["class_map"] = map_50_90_class_map
    # calculate fitness as a weighted sum of metrics, with more weight on AP at 0.5 IoU.
    weight = np.array([0.0, 0.0, 0.1, 0.9])
    map_array = np.array([
        map_50_metrics["precision"],
        map_50_metrics["recall"],
        map_50_metrics["map"],
        map_50_95_metrics["map"],
    ])
    fitness = float((map_array * weight).sum())

    return map_50_metrics, map_50_95_metrics, fitness




