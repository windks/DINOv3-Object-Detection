import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple, Union
import torch
from PIL import Image


def get_color_palette(num_classes: int) -> np.ndarray:
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors


def draw_bounding_boxes(
    image: np.ndarray,
    detections: Dict[str, torch.Tensor],
    class_names: Optional[List[str]] = None,
    colors: Optional[np.ndarray] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    img = image.copy()
    
    boxes = detections['boxes'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    
    if colors is None:
        num_classes = labels.max() + 1 if len(labels) > 0 else 1
        colors = get_color_palette(num_classes)
        
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[label].tolist()
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        text = f"{label}"
        if class_names and label < len(class_names):
            text = class_names[label]
        text += f" {score:.2f}"
        
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        cv2.rectangle(
            img,
            (x1, y1 - text_height - 4),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        cv2.putText(
            img,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        
    return img


def plot_detections(
    image: np.ndarray,
    detections: Dict[str, torch.Tensor],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.imshow(image)
    ax.axis('off')
    
    boxes = detections['boxes'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    
    colors = plt.cm.rainbow(np.linspace(0, 1, labels.max() + 1 if len(labels) > 0 else 1))
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=colors[label],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        text = f"{label}"
        if class_names and label < len(class_names):
            text = class_names[label]
        text += f" {score:.2f}"
        
        ax.text(
            x1, y1 - 5,
            text,
            fontsize=10,
            color='white',
            bbox=dict(facecolor=colors[label], alpha=0.8),
        )
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def visualize_feature_map(
    features: torch.Tensor,
    num_channels: int = 16,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    if features.dim() == 4:
        features = features[0]
        
    num_channels = min(num_channels, features.shape[0])
    
    rows = int(np.sqrt(num_channels))
    cols = int(np.ceil(num_channels / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_channels):
        feat = features[i].cpu().numpy()
        
        feat_normalized = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        
        axes[i].imshow(feat_normalized, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i}')
        
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    return fig


def save_detection_results(
    image_path: str,
    detections: Dict[str, torch.Tensor],
    output_path: str,
    class_names: Optional[List[str]] = None,
) -> None:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result_image = draw_bounding_boxes(image, detections, class_names)
    
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_image_bgr)


def create_detection_video(
    video_path: str,
    output_path: str,
    predictor,
    class_names: Optional[List[str]] = None,
    fps: Optional[int] = None,
) -> None:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps or int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detections = predictor.predict(frame_rgb, original_size=(height, width))
            
            result_frame = draw_bounding_boxes(frame_rgb, detections, class_names)
            
            result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            out.write(result_frame_bgr)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
                
    finally:
        cap.release()
        out.release()
        
    print(f"Video saved to {output_path}")