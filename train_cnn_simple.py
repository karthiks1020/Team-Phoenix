"""
Minimal CNN Training Script for Handicraft Classification
Works without problematic dependencies
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
from pathlib import Path

class HandicraftDataset(Dataset):
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
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    for img_file in class_dir.glob(ext):
                        self.images.append(str(img_file))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Create dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class HandicraftCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(HandicraftCNN, self).__init__()
        
        # Use ResNet18 for faster training
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze most layers for transfer learning
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_model():
    print("🚀 TRAINING CNN ON YOUR DATASET")
    print("=" * 50)
    
    # Check dataset
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("❌ Dataset not found!")
        return False
    
    # Count images
    categories = ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls']
    total_images = 0
    
    print("📊 Your Dataset:")
    for category in categories:
        category_path = data_dir / category
        count = 0
        if category_path.exists():
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                count += len(list(category_path.glob(ext)))
        total_images += count
        print(f"   {category}: {count} images")
    
    print(f"   Total: {total_images} images")
    
    if total_images < 20:
        print("❌ Need more images for training!")
        return False
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("\n📦 Loading your images...")
    dataset = HandicraftDataset(data_dir, transform=train_transform)
    
    if len(dataset) == 0:
        print("❌ No valid images found!")
        return False
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = min(8, max(1, len(train_dataset) // 10))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Ready: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    model = HandicraftCNN(num_classes=4)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print(f"\n🎯 Training CNN model...")
    num_epochs = 25
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validate
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
                _, predicted = torch.max(output, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate accuracy
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': best_accuracy,
                'classes': categories,
                'epoch': epoch
            }, 'models/handicraft_cnn.pth')
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}% | Best: {best_accuracy:5.1f}%")
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"🏆 Best Accuracy: {best_accuracy:.1f}%")
    print(f"💾 Model saved: models/handicraft_cnn.pth")
    
    # Save model info
    model_info = {
        'accuracy': best_accuracy,
        'classes': categories,
        'total_images': total_images,
        'trained_on': 'user_dataset'
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    if best_accuracy > 80:
        print("✅ Great performance! Ready for production.")
    elif best_accuracy > 60:
        print("👍 Good performance! Model is usable.")
    else:
        print("📈 Model trained but could improve with more data.")
    
    return True

if __name__ == "__main__":
    success = train_model()
    if success:
        print("\n🚀 CNN model trained on your dataset!")
        print("Next: Integrate into backend for real AI classification")
    else:
        print("\n❌ Training failed. Check dataset.")