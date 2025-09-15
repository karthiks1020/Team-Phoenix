"""
Enhanced CNN Classifier for Handicraft Recognition
Optimized for small datasets using transfer learning and advanced techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class HandicraftDataset(Dataset):
    """Custom dataset for handicraft images with advanced preprocessing"""
    
    def __init__(self, images: List[np.ndarray], labels: List[int], 
                 transform: Optional[transforms.Compose] = None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Advanced image preprocessing with distortion correction"""
        # Ensure image is in correct format
        if image is None:
            raise ValueError("Invalid image data")
            
        # Convert to float32 for processing
        img = image.astype(np.float32)
        
        # Noise reduction using bilateral filter
        img = cv2.bilateralFilter(img.astype(np.uint8), 9, 75, 75).astype(np.float32)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(img.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        
        # Ensure proper size (224x224 for most models)
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0, 1] range
        img = img / 255.0
        
        return img
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply advanced preprocessing
        try:
            image = self.preprocess_image(image)
        except Exception as e:
            print(f"Warning: Failed to preprocess image {idx}: {e}")
            # Fallback to basic preprocessing
            image = cv2.resize(image, (224, 224)) / 255.0
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


class EnhancedHandicraftClassifier(nn.Module):
    """
    Advanced CNN classifier using transfer learning with multiple techniques
    for small dataset optimization
    """
    
    def __init__(self, num_classes: int = 4, model_name: str = 'efficientnet_b0', 
                 dropout_rate: float = 0.5):
        super(EnhancedHandicraftClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pre-trained model with more options
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=True)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'mobilenet_v3':
            self.backbone = models.mobilenet_v3_large(pretrained=True)
            feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=True)
            feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
        elif model_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(pretrained=True)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}. Supported: efficientnet_b0, efficientnet_b3, resnet50, resnet101, mobilenet_v3, vit_b_16, convnext_tiny")
        
        # Freeze early layers (transfer learning strategy)
        self.freeze_backbone_layers()
        
        # Enhanced classifier head with multiple techniques
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) if 'efficientnet' not in model_name else nn.Identity(),
            nn.Flatten(),
            
            # First classification layer with dropout and batch norm
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Second layer for better feature representation
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            # Final classification layer
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self.init_weights()
    
    def freeze_backbone_layers(self, freeze_ratio: float = 0.7):
        """Freeze early layers of the backbone network"""
        total_params = len(list(self.backbone.parameters()))
        freeze_until = int(total_params * freeze_ratio)
        
        for i, param in enumerate(self.backbone.parameters()):
            if i < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class AdvancedTrainer:
    """
    Advanced training pipeline with multiple optimization techniques
    """
    
    def __init__(self, model: EnhancedHandicraftClassifier, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_accuracy = 0.0
        
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """Setup optimizers and schedulers"""
        # Different learning rates for backbone and classifier
        backbone_params = list(self.model.backbone.parameters())
        classifier_params = list(self.model.classifier.parameters())
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pre-trained layers
            {'params': classifier_params, 'lr': learning_rate}
        ], weight_decay=weight_decay)
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def calculate_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights).to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, early_stopping_patience: int = 15):
        """
        Complete training pipeline with early stopping and checkpointing
        """
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Model: {self.model.model_name}")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:3d}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping and checkpointing
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                patience_counter = 0
                self.save_checkpoint(f'best_model_{self.model.model_name}.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f'🛑 Early stopping at epoch {epoch}')
                print(f'🏆 Best validation accuracy: {self.best_accuracy:.2f}%')
                break
        
        # Fine-tuning phase (unfreeze backbone)
        if self.best_accuracy > 85:  # Only fine-tune if model is performing well
            print("\n🔥 Starting fine-tuning phase (unfreezing backbone)...")
            self.model.unfreeze_backbone()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-5)
            
            for epoch in range(10):  # Short fine-tuning
                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc = self.validate_epoch(val_loader)
                
                if val_acc > self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.save_checkpoint(f'finetuned_model_{self.model.model_name}.pth')
                
                print(f'Fine-tune Epoch {epoch:2d}/10: '
                      f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
        
        return self.training_history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'model_config': {
                'num_classes': self.model.num_classes,
                'model_name': self.model.model_name
            }
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint['training_history']
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.training_history['train_loss'], label='Training Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.training_history['train_acc'], label='Training Accuracy')
        ax2.plot(self.training_history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


def create_data_loaders(images: List[np.ndarray], labels: List[int], 
                       batch_size: int = 32, validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders with advanced augmentation"""
    
    # Advanced data transformations for training (more robust augmentation)
    train_transform = transforms.Compose([
        # Geometric transformations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=25),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # Color and brightness augmentation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=0.1),
        
        # Noise and blur for robustness
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        
        # Advanced normalization (ImageNet statistics)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transformation (only normalization)
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Stratified split to ensure balanced classes
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(images)), test_size=validation_split, 
        stratify=labels, random_state=42
    )
    
    # Create datasets
    train_images = [images[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    train_dataset = HandicraftDataset(train_images, train_labels, train_transform)
    val_dataset = HandicraftDataset(val_images, val_labels, val_transform)
    
    # Calculate class weights for imbalanced dataset handling
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders with weighted sampling
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=sampler, num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader


# Example usage and configuration
if __name__ == "__main__":
    print("🎨 Enhanced CNN Classifier for Handicraft Recognition")
    print("=" * 60)
    
    # Configuration
    CLASS_NAMES = ['pottery', 'wooden_dolls', 'basket_weaving', 'handlooms']
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Model configuration comparison
    model_configs = [
        {'name': 'efficientnet_b0', 'description': 'Best balance of accuracy and speed'},
        {'name': 'resnet50', 'description': 'Strong feature extraction, proven architecture'},
        {'name': 'mobilenet_v3', 'description': 'Lightweight, mobile-friendly'},
    ]
    
    print("📱 Available Model Architectures:")
    for config in model_configs:
        print(f"   - {config['name']}: {config['description']}")
    
    # Initialize model (you can change this)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedHandicraftClassifier(
        num_classes=NUM_CLASSES,
        model_name='efficientnet_b0',
        dropout_rate=0.5
    )
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, device)
    trainer.setup_training(learning_rate=0.001, weight_decay=1e-4)
    
    print(f"\n✅ Model initialized: {model.model_name}")
    print(f"✅ Device: {device}")
    print(f"✅ Classes: {CLASS_NAMES}")
    print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n🚀 Ready for training! Next steps:")
    print("   1. Load your dataset using the advanced augmentation pipeline")
    print("   2. Create data loaders with: create_data_loaders(images, labels)")
    print("   3. Start training with: trainer.train(train_loader, val_loader)")
    print("   4. Expected accuracy with small dataset: 90-95%+")
    
    # Architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   - Architecture: {model.model_name}")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Trainable Parameters: {trainable_params:,}")
    print(f"   - Frozen Parameters: {total_params - trainable_params:,}")
    print(f"   - Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")