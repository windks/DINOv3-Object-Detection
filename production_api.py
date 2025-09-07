"""
Production-ready API for DINOv3 Object Detection
Simple interface for real-world object detection tasks
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
import torch
from dataclasses import dataclass, asdict

from dinov3_od.zero_shot_detector import DINOv3ZeroShotDetector, Detection


class DINOv3DetectorAPI:
    """
    Simple API for production object detection using DINOv3.
    
    Example:
        # Initialize detector
        detector = DINOv3DetectorAPI()
        
        # Load image
        image = cv2.imread("image.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        results = detector.detect(image, ["person", "car", "dog"])
        
        # Process results
        for detection in results:
            print(f"{detection.class_name}: {detection.confidence:.2f} at {detection.bbox}")
    """
    
    def __init__(
        self,
        model_size: str = "small",  # small, base, or large
        device: str = None,
        use_fp16: bool = True,
    ):
        """
        Initialize detector.
        
        Args:
            model_size: Model size - "small", "base", or "large"
            device: Device to use (None for auto-detect)
            use_fp16: Use half precision for faster inference
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        
        # Model mapping
        model_map = {
            "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        }
        
        dinov3_model = model_map.get(model_size, model_map["small"])
        
        # Initialize detector
        self.detector = DINOv3ZeroShotDetector(
            dinov3_model=dinov3_model,
            device=self.device
        )
        
        # Convert to FP16 if requested
        if self.use_fp16:
            self.detector.dinov3_model = self.detector.dinov3_model.half()
            self.detector.clip_model = self.detector.clip_model.half()
            self.detector.feature_projector = self.detector.feature_projector.half()
    
    def detect(
        self,
        image: Union[np.ndarray, str, Path],
        targets: List[str],
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> List[Detection]:
        """
        Detect objects in image.
        
        Args:
            image: Input image (numpy array, file path, or Path object)
            targets: List of target object descriptions (e.g., ["person", "red car", "small dog"])
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for Non-Maximum Suppression
            
        Returns:
            List of Detection objects with class_name, confidence, and bbox
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Run detection
        detections = self.detector.detect(
            image=image,
            text_queries=targets,
            threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )
        
        return detections
    
    def detect_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        targets: List[str],
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> List[List[Detection]]:
        """
        Detect objects in multiple images.
        
        Args:
            images: List of images
            targets: List of target object descriptions
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for NMS
            
        Returns:
            List of detection results for each image
        """
        results = []
        for image in images:
            detections = self.detect(
                image,
                targets,
                confidence_threshold,
                nms_threshold
            )
            results.append(detections)
        return results
    
    def visualize(
        self,
        image: Union[np.ndarray, str, Path],
        detections: List[Detection],
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Visualize detection results.
        
        Args:
            image: Original image
            detections: Detection results
            save_path: Optional path to save visualization
            
        Returns:
            Image with drawn detections
        """
        # Load image if needed
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        # Use detector's visualization
        vis_image = self.detector.visualize(image, detections, save_path)
        
        return vis_image
    
    def to_json(self, detections: List[Detection]) -> str:
        """Convert detections to JSON string"""
        return json.dumps([asdict(d) for d in detections], indent=2)
    
    def to_dict(self, detections: List[Detection]) -> List[Dict]:
        """Convert detections to list of dictionaries"""
        return [asdict(d) for d in detections]
    
    def filter_detections(
        self,
        detections: List[Detection],
        min_confidence: float = None,
        class_names: List[str] = None,
        min_area: int = None,
    ) -> List[Detection]:
        """
        Filter detections based on criteria.
        
        Args:
            detections: List of detections to filter
            min_confidence: Minimum confidence threshold
            class_names: List of class names to keep
            min_area: Minimum bounding box area
            
        Returns:
            Filtered list of detections
        """
        filtered = detections
        
        if min_confidence is not None:
            filtered = [d for d in filtered if d.confidence >= min_confidence]
        
        if class_names is not None:
            filtered = [d for d in filtered if d.class_name in class_names]
        
        if min_area is not None:
            filtered = [
                d for d in filtered
                if (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) >= min_area
            ]
        
        return filtered
    
    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image from file"""
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image from {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Convenience functions for quick usage
def detect_objects(
    image_path: str,
    targets: List[str],
    threshold: float = 0.3,
    visualize: bool = False,
    save_path: str = None,
) -> List[Dict]:
    """
    Quick function to detect objects in an image.
    
    Example:
        results = detect_objects("photo.jpg", ["person", "car", "dog"])
        for obj in results:
            print(f"Found {obj['class_name']} with confidence {obj['confidence']}")
    """
    # Initialize API
    api = DINOv3DetectorAPI()
    
    # Detect objects
    detections = api.detect(image_path, targets, threshold)
    
    # Visualize if requested
    if visualize or save_path:
        api.visualize(image_path, detections, save_path)
    
    # Return as dictionaries
    return api.to_dict(detections)


def process_video(
    video_path: str,
    output_path: str,
    targets: List[str],
    threshold: float = 0.3,
    fps: int = None,
) -> None:
    """
    Process video and detect objects in each frame.
    
    Example:
        process_video("input.mp4", "output.mp4", ["person", "car"])
    """
    # Initialize API
    api = DINOv3DetectorAPI()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps or int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect objects
            detections = api.detect(frame_rgb, targets, threshold)
            
            # Visualize
            vis_frame = api.visualize(frame_rgb, detections)
            
            # Convert back to BGR and write
            out.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
    
    finally:
        cap.release()
        out.release()
    
    print(f"Video processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("DINOv3 Object Detection API Example")
    print("-" * 50)
    
    # Example 1: Simple detection
    print("Example 1: Simple object detection")
    print("""
    from production_api import detect_objects
    
    # Detect objects in image
    results = detect_objects(
        "image.jpg",
        ["person", "car", "traffic light"],
        threshold=0.3,
        visualize=True,
        save_path="result.jpg"
    )
    
    # Process results
    for obj in results:
        print(f"Found {obj['class_name']} at {obj['bbox']}")
    """)
    
    # Example 2: API usage
    print("\nExample 2: Using the API class")
    print("""
    from production_api import DINOv3DetectorAPI
    
    # Initialize detector
    detector = DINOv3DetectorAPI(model_size="base")
    
    # Detect objects
    image = cv2.imread("photo.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detections = detector.detect(
        image,
        ["red car", "blue truck", "yellow bus"],
        confidence_threshold=0.25
    )
    
    # Filter results
    high_conf = detector.filter_detections(detections, min_confidence=0.5)
    
    # Save as JSON
    json_results = detector.to_json(high_conf)
    """)
    
    # Example 3: Video processing
    print("\nExample 3: Process video")
    print("""
    from production_api import process_video
    
    # Process video
    process_video(
        "input_video.mp4",
        "output_video.mp4",
        ["person", "bicycle", "car"],
        threshold=0.3
    )
    """)