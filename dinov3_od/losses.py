import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        return loss.mean()


class SmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = torch.abs(inputs - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
            
        return loss.mean()


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        bbox_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_loss = FocalLoss(focal_loss_alpha, focal_loss_gamma)
        self.smooth_l1_loss = SmoothL1Loss()
        self.bbox_loss_weight = bbox_loss_weight
        self.cls_loss_weight = cls_loss_weight
        
        self.matcher = HungarianMatcher()
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        class_logits = outputs["class_logits"]
        box_regression = outputs["box_regression"]
        anchors = outputs["anchors"]
        
        device = class_logits.device
        batch_size = class_logits.shape[0]
        
        all_cls_losses = []
        all_bbox_losses = []
        
        for idx in range(batch_size):
            target = targets[idx]
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            
            if len(gt_boxes) == 0:
                all_cls_losses.append(torch.tensor(0.0, device=device))
                all_bbox_losses.append(torch.tensor(0.0, device=device))
                continue
            
            matched_idxs, labels = self.match_anchors_to_gt(
                anchors, gt_boxes, gt_labels
            )
            
            cls_targets = self._prepare_cls_targets(
                matched_idxs, labels, self.num_classes, device
            )
            
            cls_loss = self.focal_loss(
                class_logits[idx], cls_targets
            )
            
            pos_idx = matched_idxs >= 0
            num_pos = pos_idx.sum()
            
            if num_pos > 0:
                matched_gt_boxes = gt_boxes[matched_idxs[pos_idx]]
                bbox_targets = self.encode_boxes(
                    matched_gt_boxes, anchors[pos_idx]
                )
                
                bbox_loss = self.smooth_l1_loss(
                    box_regression[idx][pos_idx],
                    bbox_targets
                )
            else:
                bbox_loss = torch.tensor(0.0, device=device)
                
            all_cls_losses.append(cls_loss)
            all_bbox_losses.append(bbox_loss)
            
        cls_loss = torch.stack(all_cls_losses).mean()
        bbox_loss = torch.stack(all_bbox_losses).mean()
        
        total_loss = (
            self.cls_loss_weight * cls_loss +
            self.bbox_loss_weight * bbox_loss
        )
        
        return {
            "loss": total_loss,
            "classification_loss": cls_loss,
            "localization_loss": bbox_loss,
        }
    
    def match_anchors_to_gt(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        iou_threshold: float = 0.5,
        low_threshold: float = 0.4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ious = box_iou(anchors, gt_boxes)
        
        matched_vals, matched_idxs = ious.max(dim=1)
        
        below_low_threshold = matched_vals < low_threshold
        between_thresholds = (matched_vals >= low_threshold) & (matched_vals < iou_threshold)
        
        matched_idxs[below_low_threshold] = -2
        matched_idxs[between_thresholds] = -1
        
        labels = torch.zeros_like(matched_idxs)
        labels[matched_idxs >= 0] = gt_labels[matched_idxs[matched_idxs >= 0]]
        
        return matched_idxs, labels
    
    def _prepare_cls_targets(
        self,
        matched_idxs: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        cls_targets = torch.zeros(
            (len(matched_idxs), num_classes),
            dtype=torch.float32,
            device=device
        )
        
        pos_idx = matched_idxs >= 0
        cls_targets[pos_idx, labels[pos_idx]] = 1.0
        
        return cls_targets
    
    def encode_boxes(
        self,
        reference_boxes: torch.Tensor,
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
        
        gt_cx = (reference_boxes[:, 0] + reference_boxes[:, 2]) / 2
        gt_cy = (reference_boxes[:, 1] + reference_boxes[:, 3]) / 2
        gt_w = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_h = reference_boxes[:, 3] - reference_boxes[:, 1]
        
        targets_dx = wx * (gt_cx - anchors_cx) / anchors_w
        targets_dy = wy * (gt_cy - anchors_cy) / anchors_h
        targets_dw = ww * torch.log(gt_w / anchors_w)
        targets_dh = wh * torch.log(gt_h / anchors_h)
        
        targets = torch.stack([targets_dx, targets_dy, targets_dw, targets_dh], dim=1)
        
        return targets


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return []


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    iou = inter / (area1[:, None] + area2 - inter)
    
    return iou