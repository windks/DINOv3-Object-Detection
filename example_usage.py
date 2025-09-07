import torch
from dinov3_od.models.feature_extractor import DINOv3FeatureExtractor
from dinov3_od.models.detection_head import DINOv3ObjectDetector
from dinov3_od.data.dataset import CocoDetectionDataset, create_data_loader
from dinov3_od.losses import DetectionLoss
from dinov3_od.utils.inference import Predictor
from dinov3_od.utils.visualization import draw_bounding_boxes
import numpy as np
import cv2


def example_training():
    print("Example: Training DINOv3 Object Detector")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    feature_extractor = DINOv3FeatureExtractor(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        device=device,
        freeze_backbone=True,
    )
    
    model = DINOv3ObjectDetector(
        feature_extractor=feature_extractor,
        num_classes=80,
    ).to(device)
    
    criterion = DetectionLoss(num_classes=80)
    
    dummy_image = torch.randn(2, 3, 518, 518).to(device)
    dummy_targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32).to(device),
            'labels': torch.tensor([1, 2], dtype=torch.int64).to(device),
        },
        {
            'boxes': torch.tensor([[150, 150, 250, 250]], dtype=torch.float32).to(device),
            'labels': torch.tensor([0], dtype=torch.int64).to(device),
        }
    ]
    
    model.train()
    outputs = model(dummy_image, dummy_targets)
    
    print(f"Training mode output:")
    print(f"  Total loss: {outputs['loss'].item():.4f}")
    print(f"  Classification loss: {outputs['classification_loss'].item():.4f}")
    print(f"  Localization loss: {outputs['localization_loss'].item():.4f}")
    
    print("\nInference mode output:")
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_image)
        
    print(f"  Class logits shape: {outputs['class_logits'].shape}")
    print(f"  Box regression shape: {outputs['box_regression'].shape}")
    print(f"  Anchors shape: {outputs['anchors'].shape}")


def example_inference():
    print("\n\nExample: Inference with DINOv3 Object Detector")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    feature_extractor = DINOv3FeatureExtractor(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        device=device,
    )
    
    model = DINOv3ObjectDetector(
        feature_extractor=feature_extractor,
        num_classes=80,
    ).to(device)
    
    predictor = Predictor(
        model,
        feature_extractor.processor,
        device=device,
        score_threshold=0.5,
        nms_threshold=0.5,
    )
    
    dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    detections = predictor.predict(dummy_image)
    
    print(f"Detected {len(detections['boxes'])} objects")
    print(f"  Boxes shape: {detections['boxes'].shape}")
    print(f"  Scores shape: {detections['scores'].shape}")
    print(f"  Labels shape: {detections['labels'].shape}")
    
    if len(detections['boxes']) > 0:
        visualized = draw_bounding_boxes(dummy_image, detections)
        print(f"  Visualization shape: {visualized.shape}")


def example_custom_dataset():
    print("\n\nExample: Creating a Custom Dataset")
    
    annotation_example = {
        "image1.jpg": {
            "boxes": [[10, 20, 100, 150], [200, 300, 400, 500]],
            "labels": [0, 1]
        },
        "image2.jpg": {
            "boxes": [[50, 60, 200, 250]],
            "labels": [2]
        }
    }
    
    print("Custom annotation format:")
    print(json.dumps(annotation_example, indent=2))
    
    print("\nTo use custom dataset:")
    print("1. Save annotations in the format above to 'annotations.json'")
    print("2. Create dataset with: CustomDetectionDataset(image_dir, 'annotations.json')")


if __name__ == "__main__":
    import json
    
    example_training()
    example_inference()
    example_custom_dataset()
    
    print("\n\nTo train on COCO dataset:")
    print("python train.py \\")
    print("  --train-data /path/to/coco/train2017 \\")
    print("  --train-ann /path/to/coco/annotations/instances_train2017.json \\")
    print("  --num-classes 80 \\")
    print("  --batch-size 16 \\")
    print("  --num-epochs 50")
    
    print("\n\nTo run inference:")
    print("python inference.py \\")
    print("  --input /path/to/image.jpg \\")
    print("  --checkpoint outputs/checkpoints/checkpoint_epoch_49.pth \\")
    print("  --visualize \\")
    print("  --save-json")