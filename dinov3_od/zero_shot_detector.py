import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import cv2
from scipy.ndimage import label, find_objects
from dataclasses import dataclass
import warnings


@dataclass
class Detection:
    """Single detection result"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class DINOv3ZeroShotDetector:
    """
    Zero-shot object detector using DINOv3 features and CLIP for text understanding.
    Combines DINOv3's powerful visual features with CLIP's text-image alignment.
    """
    
    def __init__(
        self,
        dinov3_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        clip_model: str = "openai/clip-vit-base-patch32",
        device: str = None,
        patch_size: int = 16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        
        # Load DINOv3 for visual features
        print(f"Loading DINOv3 model: {dinov3_model}")
        self.dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_model)
        self.dinov3_model = AutoModel.from_pretrained(dinov3_model).to(self.device)
        self.dinov3_model.eval()
        
        # Load CLIP for text-image alignment
        print(f"Loading CLIP model: {clip_model}")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.clip_model.eval()
        
        # Feature dimension mapping
        self.dinov3_hidden_size = self.dinov3_model.config.hidden_size
        self.clip_hidden_size = self.clip_model.config.projection_dim
        
        # Learnable projection to align DINOv3 features with CLIP space
        self.feature_projector = nn.Linear(
            self.dinov3_hidden_size, 
            self.clip_hidden_size
        ).to(self.device)
        
        # Initialize with identity-like mapping
        with torch.no_grad():
            self.feature_projector.weight.data = torch.eye(
                self.clip_hidden_size, 
                self.dinov3_hidden_size
            )[:self.clip_hidden_size, :].to(self.device)
            self.feature_projector.bias.data.zero_()
    
    @torch.no_grad()
    def extract_dinov3_features(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """Extract spatially-aware features from DINOv3"""
        inputs = self.dinov3_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        
        outputs = self.dinov3_model(pixel_values, output_hidden_states=True)
        
        # Get features from multiple layers for richer representation
        hidden_states = outputs.hidden_states
        
        # Use last 4 layers
        features = []
        for layer_idx in [-4, -3, -2, -1]:
            layer_features = hidden_states[layer_idx][:, 1:]  # Remove CLS token
            features.append(layer_features)
        
        # Average features from multiple layers
        combined_features = torch.stack(features).mean(dim=0)
        
        # Calculate spatial dimensions
        b, num_patches, hidden_dim = combined_features.shape
        h = w = int(np.sqrt(num_patches))
        
        # Reshape to spatial format
        spatial_features = combined_features.reshape(b, h, w, hidden_dim)
        
        return {
            "features": combined_features,
            "spatial_features": spatial_features,
            "height": h,
            "width": w,
        }
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text descriptions using CLIP"""
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        text_features = self.clip_model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def compute_similarity_maps(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity between visual patches and text features"""
        # Project DINOv3 features to CLIP space
        b, h, w, d = visual_features.shape
        visual_features_flat = visual_features.reshape(b * h * w, d)
        projected_features = self.feature_projector(visual_features_flat)
        projected_features = F.normalize(projected_features, dim=-1)
        projected_features = projected_features.reshape(b, h, w, -1)
        
        # Compute similarity for each text query
        num_texts = text_features.shape[0]
        similarity_maps = []
        
        for i in range(num_texts):
            text_feat = text_features[i].unsqueeze(0)
            # Compute cosine similarity
            sim_map = torch.einsum('bhwd,td->bhw', projected_features, text_feat)
            similarity_maps.append(sim_map)
        
        return torch.stack(similarity_maps, dim=1)  # [B, num_texts, H, W]
    
    def extract_boxes_from_heatmap(
        self,
        heatmap: np.ndarray,
        original_size: Tuple[int, int],
        threshold: float = 0.5,
        min_area: int = 100,
    ) -> List[Tuple[int, int, int, int]]:
        """Extract bounding boxes from similarity heatmap"""
        h_orig, w_orig = original_size
        h_map, w_map = heatmap.shape
        
        # Threshold and binarize
        binary_map = heatmap > threshold
        
        # Find connected components
        labeled_map, num_features = label(binary_map)
        
        boxes = []
        slices = find_objects(labeled_map)
        
        for i, slice_obj in enumerate(slices):
            if slice_obj is None:
                continue
                
            # Get component mask
            component_mask = labeled_map[slice_obj] == (i + 1)
            
            # Check minimum area
            if component_mask.sum() < min_area:
                continue
            
            # Get bounding box in heatmap coordinates
            y_slice, x_slice = slice_obj
            y1, y2 = y_slice.start, y_slice.stop
            x1, x2 = x_slice.start, x_slice.stop
            
            # Convert to original image coordinates
            x1_orig = int(x1 * w_orig / w_map)
            y1_orig = int(y1 * h_orig / h_map)
            x2_orig = int(x2 * w_orig / w_map)
            y2_orig = int(y2 * h_orig / h_map)
            
            # Get confidence as mean similarity in the region
            confidence = heatmap[y1:y2, x1:x2][component_mask].mean()
            
            boxes.append((x1_orig, y1_orig, x2_orig, y2_orig, float(confidence)))
        
        return boxes
    
    def detect(
        self,
        image: np.ndarray,
        text_queries: List[str],
        threshold: float = 0.5,
        nms_threshold: float = 0.5,
        min_area: int = 100,
    ) -> List[Detection]:
        """
        Detect objects in image based on text descriptions.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            text_queries: List of text descriptions for target objects
            threshold: Similarity threshold for detection
            nms_threshold: IoU threshold for NMS
            min_area: Minimum area for valid detections
            
        Returns:
            List of Detection objects
        """
        # Extract visual features
        visual_data = self.extract_dinov3_features(image)
        spatial_features = visual_data["spatial_features"]
        
        # Encode text queries
        text_features = self.encode_text(text_queries)
        
        # Compute similarity maps
        similarity_maps = self.compute_similarity_maps(spatial_features, text_features)
        
        # Extract detections for each query
        all_detections = []
        
        for idx, (query, sim_map) in enumerate(zip(text_queries, similarity_maps[0])):
            # Convert to numpy
            sim_map_np = sim_map.cpu().numpy()
            
            # Apply Gaussian smoothing for better localization
            sim_map_smooth = cv2.GaussianBlur(sim_map_np, (5, 5), 1.0)
            
            # Extract boxes
            boxes = self.extract_boxes_from_heatmap(
                sim_map_smooth,
                image.shape[:2],
                threshold,
                min_area
            )
            
            # Create Detection objects
            for x1, y1, x2, y2, conf in boxes:
                all_detections.append(Detection(
                    class_name=query,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2)
                ))
        
        # Apply NMS
        if all_detections and nms_threshold < 1.0:
            all_detections = self.apply_nms(all_detections, nms_threshold)
        
        return all_detections
    
    def apply_nms(self, detections: List[Detection], threshold: float) -> List[Detection]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return detections
        
        # Group by class
        detections_by_class = {}
        for det in detections:
            if det.class_name not in detections_by_class:
                detections_by_class[det.class_name] = []
            detections_by_class[det.class_name].append(det)
        
        # Apply NMS per class
        final_detections = []
        
        for class_name, class_dets in detections_by_class.items():
            # Convert to arrays
            boxes = np.array([list(d.bbox) for d in class_dets])
            scores = np.array([d.confidence for d in class_dets])
            
            # Apply NMS
            indices = self._nms(boxes, scores, threshold)
            
            # Keep selected detections
            for idx in indices:
                final_detections.append(class_dets[idx])
        
        return final_detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """NumPy NMS implementation"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[Detection],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Visualize detection results"""
        vis_image = image.copy()
        
        # Color palette
        colors = plt.cm.rainbow(np.linspace(0, 1, 20))
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Get color for class
            class_idx = hash(det.class_name) % 20
            color = (colors[class_idx][:3] * 255).astype(int).tolist()
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image


# Import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib not available, some visualization features may be limited")
    plt = None