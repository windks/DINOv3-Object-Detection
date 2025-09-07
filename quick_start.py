"""
Quick Start Guide for DINOv3 Object Detection

This script demonstrates how to use pretrained DINOv3 for object detection
without any training required.
"""

import cv2
import numpy as np
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image

# Import our production API
from production_api import DINOv3DetectorAPI, detect_objects


def download_sample_image(url: str) -> np.ndarray:
    """Download a sample image from URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)


def example_1_basic_detection():
    """Example 1: Basic object detection with visualization"""
    print("=" * 60)
    print("Example 1: Basic Object Detection")
    print("=" * 60)
    
    # Initialize detector (uses pretrained weights)
    detector = DINOv3DetectorAPI(model_size="small")
    
    # Create a sample image (you can replace with your own)
    # In practice, load your image with cv2.imread()
    sample_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    # Define what you want to detect
    targets = ["person", "car", "dog", "cat", "bicycle"]
    
    # Run detection
    detections = detector.detect(
        image=sample_image,
        targets=targets,
        confidence_threshold=0.3
    )
    
    # Print results
    print(f"\nFound {len(detections)} objects:")
    for det in detections:
        print(f"  - {det.class_name}: confidence={det.confidence:.2f}, bbox={det.bbox}")
    
    # Visualize (optional)
    if detections:
        vis_image = detector.visualize(sample_image, detections)
        # cv2.imwrite("detection_result.jpg", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print("\nVisualization saved to 'detection_result.jpg'")


def example_2_real_image():
    """Example 2: Detection on a real image"""
    print("\n" + "=" * 60)
    print("Example 2: Detection on Real Image")
    print("=" * 60)
    
    # For this example, we'll create a dummy image
    # In practice, replace with: image_path = "your_image.jpg"
    image_path = "sample_image.jpg"
    
    # Create a dummy image for demonstration
    dummy_image = np.ones((800, 600, 3), dtype=np.uint8) * 128
    cv2.rectangle(dummy_image, (100, 100), (300, 400), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(dummy_image, (400, 200), (550, 350), (0, 255, 0), -1)  # Green rectangle
    cv2.imwrite(image_path, dummy_image)
    
    # Use the simple detection function
    results = detect_objects(
        image_path=image_path,
        targets=["blue object", "green object", "red object"],
        threshold=0.25,
        visualize=True,
        save_path="detected_objects.jpg"
    )
    
    print(f"\nDetection results:")
    for obj in results:
        print(f"  - {obj['class_name']}: confidence={obj['confidence']:.2f}")
        print(f"    Location: x1={obj['bbox'][0]}, y1={obj['bbox'][1]}, "
              f"x2={obj['bbox'][2]}, y2={obj['bbox'][3]}")


def example_3_custom_targets():
    """Example 3: Detection with custom descriptive targets"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Descriptive Targets")
    print("=" * 60)
    
    # Initialize detector
    detector = DINOv3DetectorAPI(model_size="base")  # Use larger model for better accuracy
    
    # Create sample scene
    scene = np.ones((600, 800, 3), dtype=np.uint8) * 200
    
    # You can use very descriptive targets
    descriptive_targets = [
        "red sports car",
        "person wearing blue shirt",
        "small brown dog",
        "yellow delivery truck",
        "green traffic light",
        "stop sign",
        "pedestrian crossing"
    ]
    
    # Run detection with descriptive queries
    detections = detector.detect(
        image=scene,
        targets=descriptive_targets,
        confidence_threshold=0.2
    )
    
    print(f"\nSearching for {len(descriptive_targets)} descriptive targets...")
    print("This demonstrates zero-shot capability - no training needed!")
    
    if detections:
        for det in detections:
            print(f"  - Found: {det.class_name} (conf: {det.confidence:.2f})")
    else:
        print("  - No objects detected in this synthetic example")


def example_4_batch_processing():
    """Example 4: Process multiple images"""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)
    
    # Initialize detector
    detector = DINOv3DetectorAPI()
    
    # Create multiple sample images
    images = []
    for i in range(3):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        images.append(img)
    
    # Define targets
    targets = ["vehicle", "person", "animal"]
    
    # Process batch
    all_results = detector.detect_batch(
        images=images,
        targets=targets,
        confidence_threshold=0.3
    )
    
    # Print results for each image
    for idx, detections in enumerate(all_results):
        print(f"\nImage {idx + 1}: Found {len(detections)} objects")
        for det in detections:
            print(f"  - {det.class_name}: {det.confidence:.2f}")


def example_5_filtering_results():
    """Example 5: Filtering detection results"""
    print("\n" + "=" * 60)
    print("Example 5: Filtering Detection Results")
    print("=" * 60)
    
    # Initialize detector
    detector = DINOv3DetectorAPI()
    
    # Dummy image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 128
    
    # Detect multiple object types
    targets = ["car", "truck", "bus", "person", "bicycle", "motorcycle"]
    
    # Run detection
    all_detections = detector.detect(image, targets, confidence_threshold=0.2)
    
    # Simulate some detections for demonstration
    from dinov3_od.zero_shot_detector import Detection
    all_detections = [
        Detection("car", 0.8, (100, 100, 200, 200)),
        Detection("person", 0.6, (300, 150, 350, 250)),
        Detection("truck", 0.3, (400, 100, 500, 200)),
        Detection("bicycle", 0.7, (50, 300, 150, 400)),
        Detection("car", 0.4, (500, 300, 600, 400)),
    ]
    
    print(f"Total detections: {len(all_detections)}")
    
    # Filter by confidence
    high_conf = detector.filter_detections(all_detections, min_confidence=0.5)
    print(f"\nHigh confidence (>0.5): {len(high_conf)} detections")
    for det in high_conf:
        print(f"  - {det.class_name}: {det.confidence:.2f}")
    
    # Filter by class
    vehicles_only = detector.filter_detections(
        all_detections, 
        class_names=["car", "truck", "bus", "motorcycle"]
    )
    print(f"\nVehicles only: {len(vehicles_only)} detections")
    for det in vehicles_only:
        print(f"  - {det.class_name}: {det.confidence:.2f}")
    
    # Filter by area
    large_objects = detector.filter_detections(all_detections, min_area=5000)
    print(f"\nLarge objects (area > 5000): {len(large_objects)} detections")


def main():
    """Run all examples"""
    print("DINOv3 Object Detection - Quick Start Examples")
    print("=" * 60)
    print("\nThis demonstrates using pretrained DINOv3 for object detection")
    print("No training required - works out of the box!")
    print("\nFeatures:")
    print("- Zero-shot detection: Detect any object by description")
    print("- No training needed: Uses pretrained DINOv3 + CLIP")
    print("- Flexible targets: Use simple or descriptive text queries")
    print("- Production ready: Simple API for easy integration")
    
    # Run examples
    example_1_basic_detection()
    example_2_real_image()
    example_3_custom_targets()
    example_4_batch_processing()
    example_5_filtering_results()
    
    print("\n" + "=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print("\nTo use in your own project:")
    print("""
    from production_api import DINOv3DetectorAPI
    
    # Initialize
    detector = DINOv3DetectorAPI()
    
    # Detect
    results = detector.detect(
        "your_image.jpg",
        ["person", "car", "dog"],
        confidence_threshold=0.3
    )
    
    # Process results
    for detection in results:
        print(f"{detection.class_name}: {detection.bbox}")
    """)
    
    # Cleanup
    import os
    for file in ["sample_image.jpg", "detected_objects.jpg"]:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    main()