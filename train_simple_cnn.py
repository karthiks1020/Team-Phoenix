"""
Simplified CNN Training Script for Handicraft Classification
Works with your organized dataset and handles dependency issues
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import random

class HandicraftDataset(Dataset):
    """Simple dataset for handicraft images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all images and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob("*.jpg"):
                    self.images.append(str(img_file))
                    self.labels.append(self.class_to_idx[class_name])
                for img_file in class_dir.glob("*.png"):
                    self.images.append(str(img_file))
                    self.labels.append(self.class_to_idx[class_name])
                for img_file in class_dir.glob("*.jpeg"):
                    self.images.append(str(img_file))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image with OpenCV
        image = cv2.imread(img_path)
        if image is None:
            # Fallback: create a dummy image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
        
        # Convert to PIL format for transforms
        from PIL import Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleHandicraftCNN(nn.Module):
    """Simple CNN using transfer learning"""
    
    def __init__(self, num_classes=4):
        super(SimpleHandicraftCNN, self).__init__()
        
        # Use ResNet18 as backbone (lighter than ResNet50)
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_cnn_model():
    """Train CNN model with your dataset"""
    
    print("🚀 STARTING CNN TRAINING")
    print("=" * 50)
    
    # Check if dataset exists
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("❌ Dataset directory not found!")
        return
    
    # Count images per category
    categories = ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls']
    total_images = 0
    
    print("📊 Dataset Statistics:")
    for category in categories:
        category_path = data_dir / category
        if category_path.exists():
            count = len(list(category_path.glob("*.jpg"))) + \
                   len(list(category_path.glob("*.png"))) + \
                   len(list(category_path.glob("*.jpeg")))
            total_images += count
            print(f"   {category}: {count} images")
        else:
            print(f"   {category}: 0 images (missing)")
    
    print(f"   Total: {total_images} images")
    
    if total_images < 20:
        print("❌ Not enough images for training!")
        return
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("\n📦 Loading dataset...")
    full_dataset = HandicraftDataset(data_dir, transform=train_transform)
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    batch_size = min(16, len(train_dataset) // 4)  # Adaptive batch size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    model = SimpleHandicraftCNN(num_classes=4)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    print(f"\n🎯 Starting training...")
    num_epochs = 30
    best_accuracy = 0.0
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'classes': categories
            }, 'models/best_handicraft_model.pth')
        
        # Store history
        training_history['train_loss'].append(train_loss / len(train_loader))
        training_history['train_acc'].append(train_accuracy)
        training_history['val_loss'].append(val_loss / len(val_loader))
        training_history['val_acc'].append(val_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: {train_accuracy:6.2f}% | "
              f"Val: {val_accuracy:6.2f}% | "
              f"Best: {best_accuracy:6.2f}%")
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"🏆 Best Validation Accuracy: {best_accuracy:.2f}%")
    
    # Save training history
    os.makedirs('models', exist_ok=True)
    with open('models/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"💾 Model saved: models/best_handicraft_model.pth")
    print(f"📊 Training history: models/training_history.json")
    
    # Performance analysis
    if best_accuracy > 90:
        print("🌟 Excellent performance! Ready for production.")
    elif best_accuracy > 80:
        print("✅ Good performance! Model is ready to use.")
    elif best_accuracy > 70:
        print("👍 Decent performance. Consider adding more data.")
    else:
        print("📈 Room for improvement. Try adding more diverse images.")
    
    return best_accuracy

if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Run training
    accuracy = train_cnn_model()
    
    print(f"\n🎯 Final Result: {accuracy:.2f}% accuracy")
    print("🚀 Your CNN model is now trained and ready!")
    print("📱 The backend will now use this trained model instead of simple heuristics.")