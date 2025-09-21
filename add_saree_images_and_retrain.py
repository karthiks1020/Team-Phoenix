"""
Add Saree Images and Retrain CNN Script
This script helps you add more saree images to the handlooms dataset and retrains the CNN
"""

import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import json
from datetime import datetime
import random
from PIL import Image

def check_current_dataset():
    """Check current dataset statistics"""
    print("📊 CURRENT DATASET STATUS")
    print("=" * 40)
    
    data_dir = Path("data/raw")
    categories = ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls']
    total_images = 0
    
    for category in categories:
        category_path = data_dir / category
        if category_path.exists():
            count = len(list(category_path.glob("*.jpg"))) + \
                   len(list(category_path.glob("*.png"))) + \
                   len(list(category_path.glob("*.jpeg")))
            total_images += count
            print(f"   {category:15}: {count:3d} images")
        else:
            print(f"   {category:15}: 0 images (missing)")
    
    print(f"   {'Total':15}: {total_images:3d} images")
    return total_images

def instructions_for_adding_saree_images():
    """Provide instructions for manually adding saree images"""
    print("\n📋 HOW TO ADD SAREE IMAGES")
    print("=" * 40)
    print("1. 🔍 Find saree images online or from your collection")
    print("   • Search for: 'traditional saree', 'handloom saree', 'silk saree'")
    print("   • Good sources: craft websites, textile museums, artisan portfolios")
    print("   • Recommended: 20-50 additional saree images")
    
    print("\n2. 💾 Save images to the handlooms folder:")
    print(f"   📁 Folder: data/raw/handlooms/")
    print("   📝 Naming: saree_001.jpg, saree_002.jpg, etc.")
    print("   📏 Format: JPG, PNG, or JPEG")
    print("   📐 Quality: Clear, well-lit images")
    
    print("\n3. 🎯 Image variety (for better training):")
    print("   • Different saree colors (red, blue, green, gold)")
    print("   • Different patterns (floral, geometric, traditional)")
    print("   • Different angles (front view, detailed shots)")
    print("   • Different lighting conditions")
    print("   • Traditional handloom textures")
    
    print("\n4. ▶️ After adding images, run this script again")
    
    # Wait for user to add images
    input("\n⏳ Press Enter after you've added saree images to continue...")

class HandicraftDataset(Dataset):
    """Dataset class for handicraft images"""
    
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
                # Include all image formats
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_file in class_dir.glob(ext):
                        self.images.append(str(img_file))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image with error handling
        try:
            image = cv2.imread(img_path)
            if image is None:
                # Fallback: try PIL
                pil_img = Image.open(img_path)
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = Image.fromarray(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Create dummy image
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ImprovedHandicraftCNN(nn.Module):
    """Improved CNN with better architecture"""
    
    def __init__(self, num_classes=4):
        super(ImprovedHandicraftCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers (less aggressive)
        for i, param in enumerate(self.backbone.parameters()):
            if i < 30:  # Freeze first 30 layers
                param.requires_grad = False
        
        # Improved classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_improved_cnn():
    """Train improved CNN with enhanced data"""
    print("\n🚀 TRAINING IMPROVED CNN MODEL")
    print("=" * 50)
    
    # Check dataset again
    total_images = check_current_dataset()
    
    if total_images < 50:
        print("⚠️  Dataset still small, but proceeding with training...")
    
    # Enhanced data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # Add vertical flip
        transforms.RandomRotation(degrees=20),  # More rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("📦 Loading enhanced dataset...")
    full_dataset = HandicraftDataset("data/raw", transform=train_transform)
    
    if len(full_dataset) == 0:
        print("❌ No images found in dataset!")
        return
    
    # Split dataset (85% train, 15% validation for better training)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Set validation transform
    val_dataset.dataset.transform = val_transform
    
    # Adaptive batch size
    batch_size = min(32, max(8, len(train_dataset) // 8))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} validation")
    print(f"📦 Batch size: {batch_size}")
    
    # Create improved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    model = ImprovedHandicraftCNN(num_classes=4)
    model = model.to(device)
    
    # Enhanced training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training loop
    print(f"\n🎯 Starting enhanced training...")
    num_epochs = 50  # More epochs
    best_accuracy = 0.0
    patience = 10
    no_improve_count = 0
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping and best model saving
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improve_count = 0
            
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'classes': ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls']
            }, 'models/handicraft_cnn.pth')
        else:
            no_improve_count += 1
        
        # Store history
        training_history['train_loss'].append(train_loss / len(train_loader))
        training_history['train_acc'].append(train_accuracy)
        training_history['val_loss'].append(val_loss / len(val_loader))
        training_history['val_acc'].append(val_accuracy)
        training_history['learning_rates'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: {train_accuracy:6.2f}% | "
              f"Val: {val_accuracy:6.2f}% | "
              f"Best: {best_accuracy:6.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"🏆 Best Validation Accuracy: {best_accuracy:.2f}%")
    
    # Save training history and model info
    with open('models/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    with open('models/model_info.json', 'w') as f:
        json.dump({
            'accuracy': best_accuracy,
            'classes': ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls'],
            'total_images': len(full_dataset),
            'trained_on': 'enhanced_dataset_with_sarees',
            'training_date': datetime.now().isoformat(),
            'epochs_trained': epoch + 1,
            'architecture': 'ImprovedHandicraftCNN_ResNet18'
        }, f, indent=2)
    
    print(f"💾 Model saved: models/handicraft_cnn.pth")
    print(f"📊 Training history: models/training_history.json")
    print(f"ℹ️ Model info: models/model_info.json")
    
    # Performance analysis
    if best_accuracy > 95:
        print("🌟 Excellent performance! Production ready.")
    elif best_accuracy > 85:
        print("✅ Very good performance! Ready for use.")
    elif best_accuracy > 75:
        print("👍 Good performance! Model is functional.")
    else:
        print("📈 Decent performance. Consider adding more diverse data.")
    
    return best_accuracy

def test_new_model():
    """Test the newly trained model"""
    print(f"\n🧪 TESTING NEW MODEL")
    print("=" * 30)
    
    try:
        # Import debug script functionality
        from debug_cnn_predictions import load_model, test_with_sample_images
        
        model, classes = load_model()
        if model is not None:
            test_with_sample_images(model, classes)
        else:
            print("❌ Could not load model for testing")
    except Exception as e:
        print(f"⚠️ Testing failed: {e}")
        print("💡 You can manually run: python debug_cnn_predictions.py")

def main():
    """Main function"""
    print("🎯 SAREE DATASET ENHANCEMENT & CNN RETRAINING")
    print("=" * 60)
    
    # Check current status
    initial_count = check_current_dataset()
    
    # Check if handlooms category has enough diversity
    handlooms_path = Path("data/raw/handlooms")
    handlooms_count = len(list(handlooms_path.glob("*"))) if handlooms_path.exists() else 0
    
    print(f"\n📋 CURRENT HANDLOOMS COUNT: {handlooms_count}")
    
    if handlooms_count < 50:
        print("💡 RECOMMENDATION: Add 20-30 saree images for better diversity")
        instructions_for_adding_saree_images()
    
    # Check if new images were added
    new_count = check_current_dataset()
    
    if new_count > initial_count:
        print(f"\n✅ Great! {new_count - initial_count} new images detected")
        print("🚀 Starting enhanced CNN training...")
        
        # Train improved model
        final_accuracy = train_improved_cnn()
        
        # Test new model
        test_new_model()
        
        print(f"\n🎊 ENHANCEMENT COMPLETE!")
        print(f"📈 Final accuracy: {final_accuracy:.2f}%")
        print("🔄 Backend will automatically use the new model")
        
    else:
        print(f"\n⚠️ No new images detected")
        print("🤔 You can still retrain with current data if you want:")
        
        retrain = input("   Retrain anyway? (y/N): ").lower().strip()
        if retrain == 'y':
            train_improved_cnn()
            test_new_model()
        else:
            print("👍 Okay! Add more saree images and run again.")

if __name__ == "__main__":
    main()