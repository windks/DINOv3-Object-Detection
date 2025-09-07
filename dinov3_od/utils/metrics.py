import torch
import numpy as np
from typing import Dict, List, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
import os


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def evaluate_detections(
    predictions: List[Dict[str, torch.Tensor]],
    ground_truths: List[Dict[str, torch.Tensor]],
    iou_thresholds: List[float] = None,
    num_classes: int = None,
) -> Dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))
        
    if num_classes is None:
        all_labels = []
        for gt in ground_truths:
            if len(gt['labels']) > 0:
                all_labels.extend(gt['labels'].tolist())
        num_classes = max(all_labels) + 1 if all_labels else 1
        
    ap_per_class = {cls_id: [] for cls_id in range(num_classes)}
    
    for cls_id in range(num_classes):
        for iou_threshold in iou_thresholds:
            ap = compute_class_ap(predictions, ground_truths, cls_id, iou_threshold)
            ap_per_class[cls_id].append(ap)
            
    mAP = {}
    for i, iou_threshold in enumerate(iou_thresholds):
        aps = [ap_per_class[cls_id][i] for cls_id in range(num_classes)]
        mAP[f'mAP@{iou_threshold:.2f}'] = np.mean(aps)
        
    mAP['mAP@[.5:.95]'] = np.mean([ap for aps in ap_per_class.values() for ap in aps])
    
    return mAP


def compute_class_ap(
    predictions: List[Dict[str, torch.Tensor]],
    ground_truths: List[Dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
) -> float:
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    
    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_mask = pred['labels'] == class_id
        pred_boxes = pred['boxes'][pred_mask]
        pred_scores = pred['scores'][pred_mask]
        
        gt_mask = gt['labels'] == class_id
        gt_boxes = gt['boxes'][gt_mask]
        
        for box, score in zip(pred_boxes, pred_scores):
            all_pred_boxes.append((img_idx, box))
            all_pred_scores.append(score)
            
        for box in gt_boxes:
            all_gt_boxes.append((img_idx, box))
            
    if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
        return 0.0
        
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    
    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))
    
    matched_gt = set()
    
    for pred_idx in sorted_indices:
        img_idx, pred_box = all_pred_boxes[pred_idx]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, (gt_img_idx, gt_box) in enumerate(all_gt_boxes):
            if gt_img_idx != img_idx or gt_idx in matched_gt:
                continue
                
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[pred_idx] = 1
            
    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)
    
    recalls = cumsum_tp / len(all_gt_boxes)
    precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-6)
    
    ap = compute_ap(recalls, precisions)
    
    return ap


class COCOEvaluator:
    def __init__(self, coco_gt: COCO):
        self.coco_gt = coco_gt
        self.predictions = []
        
    def update(
        self,
        image_ids: List[int],
        predictions: List[Dict[str, torch.Tensor]],
    ):
        for img_id, pred in zip(image_ids, predictions):
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                self.predictions.append({
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    'score': float(score),
                })
                
    def evaluate(self) -> Dict[str, float]:
        if len(self.predictions) == 0:
            return {'mAP': 0.0}
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.predictions, f)
            temp_file = f.name
            
        try:
            coco_dt = self.coco_gt.loadRes(temp_file)
            
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            metrics = {
                'mAP': coco_eval.stats[0],
                'mAP@.50': coco_eval.stats[1],
                'mAP@.75': coco_eval.stats[2],
                'mAP_small': coco_eval.stats[3],
                'mAP_medium': coco_eval.stats[4],
                'mAP_large': coco_eval.stats[5],
                'AR@1': coco_eval.stats[6],
                'AR@10': coco_eval.stats[7],
                'AR@100': coco_eval.stats[8],
                'AR_small': coco_eval.stats[9],
                'AR_medium': coco_eval.stats[10],
                'AR_large': coco_eval.stats[11],
            }
            
            return metrics
            
        finally:
            os.unlink(temp_file)