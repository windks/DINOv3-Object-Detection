import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from PIL import Image


def decode_boxes(
    box_regression: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    wx = 10.0
    wy = 10.0
    ww = 5.0
    wh = 5.0
    
    anchors_x1 = anchors[:, 0]
    anchors_y1 = anchors[:, 1]
    anchors_x2 = anchors[:, 2]
    anchors_y2 = anchors[:, 3]
    
    anchors_cx = (anchors_x1 + anchors_x2) / 2
    anchors_cy = (anchors_y1 + anchors_y2) / 2
    anchors_w = anchors_x2 - anchors_x1
    anchors_h = anchors_y2 - anchors_y1
    
    dx = box_regression[:, 0] / wx
    dy = box_regression[:, 1] / wy
    dw = box_regression[:, 2] / ww
    dh = box_regression[:, 3] / wh
    
    pred_cx = dx * anchors_w + anchors_cx
    pred_cy = dy * anchors_h + anchors_cy
    pred_w = torch.exp(dw) * anchors_w
    pred_h = torch.exp(dh) * anchors_h
    
    pred_x1 = pred_cx - pred_w / 2
    pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2
    pred_y2 = pred_cy + pred_h / 2
    
    pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
    
    return pred_boxes


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        idx = order[0]
        keep.append(idx)
        
        if order.numel() == 1:
            break
            
        xx1 = x1[order[1:]].clamp(min=x1[idx])
        yy1 = y1[order[1:]].clamp(min=y1[idx])
        xx2 = x2[order[1:]].clamp(max=x2[idx])
        yy2 = y2[order[1:]].clamp(max=y2[idx])
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        iou = inter / (areas[idx] + areas[order[1:]] - inter)
        
        mask = iou <= iou_threshold
        order = order[1:][mask]
        
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def postprocess_detections(
    outputs: Dict[str, torch.Tensor],
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    max_detections: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    class_logits = outputs["class_logits"]
    box_regression = outputs["box_regression"]
    anchors = outputs["anchors"]
    
    batch_size = class_logits.shape[0]
    results = []
    
    for idx in range(batch_size):
        scores = torch.sigmoid(class_logits[idx])
        
        boxes = decode_boxes(box_regression[idx], anchors)
        
        boxes[:, 0::2].clamp_(min=0)
        boxes[:, 1::2].clamp_(min=0)
        
        detections = []
        for class_id in range(scores.shape[1]):
            class_scores = scores[:, class_id]
            
            keep = class_scores > score_threshold
            class_scores = class_scores[keep]
            class_boxes = boxes[keep]
            
            if len(class_scores) > 0:
                keep_idx = nms(class_boxes, class_scores, nms_threshold)
                class_scores = class_scores[keep_idx]
                class_boxes = class_boxes[keep_idx]
                
                if len(class_scores) > max_detections:
                    top_idx = class_scores.argsort(descending=True)[:max_detections]
                    class_scores = class_scores[top_idx]
                    class_boxes = class_boxes[top_idx]
                    
                class_ids = torch.full_like(class_scores, class_id, dtype=torch.int64)
                
                detections.append({
                    'boxes': class_boxes,
                    'scores': class_scores,
                    'labels': class_ids,
                })
                
        if detections:
            all_boxes = torch.cat([d['boxes'] for d in detections])
            all_scores = torch.cat([d['scores'] for d in detections])
            all_labels = torch.cat([d['labels'] for d in detections])
            
            top_idx = all_scores.argsort(descending=True)[:max_detections]
            
            results.append({
                'boxes': all_boxes[top_idx],
                'scores': all_scores[top_idx],
                'labels': all_labels[top_idx],
            })
        else:
            results.append({
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.int64),
            })
            
    return results


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ):
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if original_size is None:
            original_size = (image.shape[0], image.shape[1])
            
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        
        outputs = self.model(pixel_values)
        
        detections = postprocess_detections(
            outputs,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )[0]
        
        processed_h, processed_w = pixel_values.shape[-2:]
        orig_h, orig_w = original_size
        
        scale_x = orig_w / processed_w
        scale_y = orig_h / processed_h
        
        if detections['boxes'].numel() > 0:
            detections['boxes'][:, 0::2] *= scale_x
            detections['boxes'][:, 1::2] *= scale_y
            
        return detections
    
    def predict_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Dict[str, torch.Tensor]]:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        
        outputs = self.model(pixel_values)
        
        detections = postprocess_detections(
            outputs,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )
        
        return detections