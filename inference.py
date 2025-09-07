import torch
import argparse
import os
import cv2
import json
from pathlib import Path
import numpy as np

from dinov3_od.models.feature_extractor import DINOv3FeatureExtractor
from dinov3_od.models.detection_head import DINOv3ObjectDetector
from dinov3_od.utils.inference import Predictor
from dinov3_od.utils.visualization import (
    draw_bounding_boxes, 
    save_detection_results,
    create_detection_video
)


def parse_args():
    parser = argparse.ArgumentParser(description='DINOv3 Object Detection Inference')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image, video, or directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                        help='Directory to save output results')
    
    parser.add_argument('--model-name', type=str, 
                        default='facebook/dinov3-vits16-pretrain-lvd1689m',
                        help='DINOv3 model name from HuggingFace')
    parser.add_argument('--class-names', type=str, default=None,
                        help='Path to JSON file with class names')
    
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Score threshold for detections')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                        help='NMS threshold for detections')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize detection results')
    parser.add_argument('--save-json', action='store_true',
                        help='Save detection results as JSON')
    
    return parser.parse_args()


def load_class_names(path: str) -> list:
    with open(path, 'r') as f:
        class_names = json.load(f)
    return class_names


def load_model(args):
    print(f"Loading model from {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    feature_extractor = DINOv3FeatureExtractor(
        model_name=args.model_name,
        freeze_backbone=True,
    )
    
    model = DINOv3ObjectDetector(
        feature_extractor=feature_extractor,
        num_classes=checkpoint['args'].num_classes,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    predictor = Predictor(
        model,
        feature_extractor.processor,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    
    return predictor


def process_image(predictor, image_path, output_dir, class_names=None, args=None):
    print(f"Processing image: {image_path}")
    
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detections = predictor.predict(image_rgb)
    
    base_name = Path(image_path).stem
    
    if args.visualize:
        result_image = draw_bounding_boxes(image_rgb, detections, class_names)
        output_path = output_dir / f"{base_name}_detections.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {output_path}")
        
    if args.save_json:
        json_output = {
            'image': str(image_path),
            'detections': {
                'boxes': detections['boxes'].tolist(),
                'scores': detections['scores'].tolist(),
                'labels': detections['labels'].tolist(),
            }
        }
        json_path = output_dir / f"{base_name}_detections.json"
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved JSON to {json_path}")
        
    return detections


def process_video(predictor, video_path, output_dir, class_names=None, args=None):
    print(f"Processing video: {video_path}")
    
    output_path = output_dir / f"{Path(video_path).stem}_detections.mp4"
    
    create_detection_video(
        str(video_path),
        str(output_path),
        predictor,
        class_names=class_names,
    )
    
    print(f"Saved video to {output_path}")


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)
    
    predictor = load_model(args)
    
    class_names = None
    if args.class_names:
        class_names = load_class_names(args.class_names)
        
    input_path = Path(args.input)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            process_image(predictor, input_path, output_dir, class_names, args)
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            process_video(predictor, input_path, output_dir, class_names, args)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
            
    elif input_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
        print(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            process_image(predictor, image_file, output_dir, class_names, args)
            
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
        
    print("Inference completed!")


if __name__ == '__main__':
    main()