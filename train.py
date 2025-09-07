import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
from tqdm import tqdm
from typing import Dict, Optional
import json

from dinov3_od.models.feature_extractor import DINOv3FeatureExtractor
from dinov3_od.models.detection_head import DINOv3ObjectDetector
from dinov3_od.data.dataset import CocoDetectionDataset, create_data_loader
from dinov3_od.losses import DetectionLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train DINOv3 Object Detection')
    
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--train-ann', type=str, required=True,
                        help='Path to training annotations file')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Path to validation data directory')
    parser.add_argument('--val-ann', type=str, default=None,
                        help='Path to validation annotations file')
    
    parser.add_argument('--model-name', type=str, 
                        default='facebook/dinov3-vits16-pretrain-lvd1689m',
                        help='DINOv3 model name from HuggingFace')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of object classes')
    parser.add_argument('--image-size', type=int, default=518,
                        help='Input image size')
    
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval (iterations)')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Model saving interval (epochs)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze DINOv3 backbone weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.setup_directories()
        self.setup_model()
        self.setup_data()
        self.setup_training()
        self.setup_logging()
        
    def setup_directories(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, 'logs'), exist_ok=True)
        
    def setup_model(self):
        print(f"Loading DINOv3 model: {self.args.model_name}")
        feature_extractor = DINOv3FeatureExtractor(
            model_name=self.args.model_name,
            device=self.device,
            freeze_backbone=self.args.freeze_backbone,
        )
        
        self.model = DINOv3ObjectDetector(
            feature_extractor=feature_extractor,
            num_classes=self.args.num_classes,
        ).to(self.device)
        
        self.processor = feature_extractor.processor
        
    def setup_data(self):
        print("Setting up datasets...")
        self.train_dataset = CocoDetectionDataset(
            root_dir=self.args.train_data,
            ann_file=self.args.train_ann,
            processor=self.processor,
            image_size=self.args.image_size,
        )
        
        self.train_loader = create_data_loader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        
        if self.args.val_data and self.args.val_ann:
            self.val_dataset = CocoDetectionDataset(
                root_dir=self.args.val_data,
                ann_file=self.args.val_ann,
                processor=self.processor,
                image_size=self.args.image_size,
            )
            
            self.val_loader = create_data_loader(
                self.val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
            )
        else:
            self.val_loader = None
            
    def setup_training(self):
        self.criterion = DetectionLoss(num_classes=self.args.num_classes)
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.num_epochs * len(self.train_loader),
            eta_min=self.args.lr * 0.01,
        )
        
        self.start_epoch = 0
        if self.args.resume:
            self.load_checkpoint(self.args.resume)
            
    def setup_logging(self):
        self.writer = SummaryWriter(os.path.join(self.args.output_dir, 'logs'))
        self.global_step = 0
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.num_epochs}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images, targets)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_cls_loss += outputs['classification_loss'].item()
            total_bbox_loss += outputs['localization_loss'].item()
            
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/cls_loss', outputs['classification_loss'].item(), self.global_step)
                self.writer.add_scalar('train/bbox_loss', outputs['localization_loss'].item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls': f'{outputs["classification_loss"].item():.4f}',
                    'bbox': f'{outputs["localization_loss"].item():.4f}',
                })
                
            self.global_step += 1
            
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'cls_loss': total_cls_loss / n_batches,
            'bbox_loss': total_bbox_loss / n_batches,
        }
    
    def validate(self) -> Optional[Dict[str, float]]:
        if self.val_loader is None:
            return None
            
        self.model.eval()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                outputs = self.model(images, targets)
                
                total_loss += outputs['loss'].item()
                total_cls_loss += outputs['classification_loss'].item()
                total_bbox_loss += outputs['localization_loss'].item()
                
        n_batches = len(self.val_loader)
        return {
            'loss': total_loss / n_batches,
            'cls_loss': total_cls_loss / n_batches,
            'bbox_loss': total_bbox_loss / n_batches,
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'args': self.args,
        }
        
        checkpoint_path = os.path.join(
            self.args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        with open(os.path.join(self.args.output_dir, 'checkpoints', 'latest.txt'), 'w') as f:
            f.write(checkpoint_path)
            
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
    def train(self):
        print(f"Starting training on {self.device}")
        
        for epoch in range(self.start_epoch, self.args.num_epochs):
            train_metrics = self.train_epoch(epoch)
            
            print(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                  f"Cls Loss: {train_metrics['cls_loss']:.4f}, "
                  f"BBox Loss: {train_metrics['bbox_loss']:.4f}")
            
            if self.val_loader:
                val_metrics = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                      f"Cls Loss: {val_metrics['cls_loss']:.4f}, "
                      f"BBox Loss: {val_metrics['bbox_loss']:.4f}")
                
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
            else:
                val_metrics = None
                
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, {'train': train_metrics, 'val': val_metrics})
                
        self.writer.close()
        print("Training completed!")


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()