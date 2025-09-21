"""
Debug CNN Model Predictions
Test the trained model with sample images to identify the handlooms bias
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path

class HandicraftCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(HandicraftCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_model():
    """Load the trained CNN model"""
    device = torch.device('cpu')
    model = HandicraftCNN(num_classes=4)
    
    model_path = 'models/handicraft_cnn.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        classes = checkpoint.get('classes', ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls'])
        accuracy = checkpoint.get('accuracy', 'Unknown')
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Training accuracy: {accuracy:.2f}%")
        print(f"🏷️ Classes: {classes}")
        
        return model, classes
    else:
        print(f"❌ Model file not found: {model_path}")
        return None, None

def test_with_sample_images(model, classes):
    """Test model with sample images from training data"""
    
    # Image preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    category_mapping = {
        'basket_weaving': 'Basket Weaving',
        'handlooms': 'Handlooms', 
        'pottery': 'Pottery',
        'wooden_dolls': 'Wooden Dolls'
    }
    
    data_dir = Path("data/raw")
    
    print("\n🔍 TESTING MODEL WITH SAMPLE IMAGES:")
    print("=" * 50)
    
    total_tests = 0
    correct_predictions = 0
    category_stats = {}
    
    for category_folder in data_dir.iterdir():
        if category_folder.is_dir() and category_folder.name in classes:
            true_category = category_folder.name
            display_category = category_mapping[true_category]
            
            print(f"\n📂 Testing {display_category} images:")
            
            # Get first 5 images from this category
            image_files = list(category_folder.glob("*.jpg")) + list(category_folder.glob("*.png"))
            test_images = image_files[:5]
            
            category_correct = 0
            category_total = 0
            predictions_for_category = []
            
            for img_file in test_images:
                try:
                    # Load and preprocess image
                    image = Image.open(img_file).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    
                    # Get prediction results
                    predicted_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_idx].item()
                    predicted_category = classes[predicted_idx]
                    predicted_display = category_mapping[predicted_category]
                    
                    # Check if correct
                    is_correct = predicted_category == true_category
                    if is_correct:
                        category_correct += 1
                        correct_predictions += 1
                    
                    category_total += 1
                    total_tests += 1
                    
                    # Store prediction info
                    predictions_for_category.append({
                        'predicted': predicted_display,
                        'confidence': confidence,
                        'correct': is_correct
                    })
                    
                    # Print detailed results
                    status = "✅" if is_correct else "❌"
                    print(f"   {status} {img_file.name[:20]:20} → {predicted_display:15} ({confidence:.2f})")
                    
                    # Print all class probabilities for debugging
                    print(f"      Raw probabilities:")
                    for i, class_name in enumerate(classes):
                        prob = probabilities[i].item()
                        display_name = category_mapping[class_name]
                        print(f"        {display_name:15}: {prob:.3f}")
                    print()
                    
                except Exception as e:
                    print(f"   ❌ Error processing {img_file.name}: {e}")
            
            # Category summary
            category_accuracy = (category_correct / category_total * 100) if category_total > 0 else 0
            category_stats[display_category] = {
                'correct': category_correct,
                'total': category_total,
                'accuracy': category_accuracy
            }
            
            print(f"   📊 Category accuracy: {category_correct}/{category_total} ({category_accuracy:.1f}%)")
    
    # Overall summary
    print(f"\n📈 OVERALL TEST RESULTS:")
    print("=" * 30)
    overall_accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
    print(f"🎯 Overall accuracy: {correct_predictions}/{total_tests} ({overall_accuracy:.1f}%)")
    
    print(f"\n📊 Per-category breakdown:")
    for category, stats in category_stats.items():
        print(f"   {category:15}: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")
    
    # Identify potential issues
    print(f"\n🔍 DIAGNOSIS:")
    handloom_stats = category_stats.get('Handlooms', {})
    if handloom_stats.get('accuracy', 0) > 80:
        print("   ⚠️ Handlooms category shows high accuracy - may be over-represented")
        
    low_accuracy_categories = [cat for cat, stats in category_stats.items() if stats.get('accuracy', 0) < 50]
    if low_accuracy_categories:
        print(f"   ⚠️ Low accuracy categories: {', '.join(low_accuracy_categories)}")
    
    if overall_accuracy < 70:
        print("   ❌ Model performance is below expected - may need retraining")
        print("   💡 Suggestions:")
        print("      • Check training data balance")
        print("      • Verify image preprocessing")
        print("      • Consider model architecture changes")

def analyze_training_data_distribution():
    """Analyze the distribution of training data"""
    print(f"\n📊 TRAINING DATA ANALYSIS:")
    print("=" * 30)
    
    data_dir = Path("data/raw")
    categories = ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls']
    
    total_images = 0
    distribution = {}
    
    for category in categories:
        category_path = data_dir / category
        if category_path.exists():
            image_count = len(list(category_path.glob("*.jpg"))) + \
                         len(list(category_path.glob("*.png"))) + \
                         len(list(category_path.glob("*.jpeg")))
            total_images += image_count
            distribution[category] = image_count
        else:
            distribution[category] = 0
    
    print(f"Total images: {total_images}")
    for category, count in distribution.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"   {category:15}: {count:3d} images ({percentage:5.1f}%)")
    
    # Check for data imbalance
    if total_images > 0:
        max_count = max(distribution.values())
        min_count = min(distribution.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\n🔍 Data balance analysis:")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            print("   ⚠️ Significant data imbalance detected!")
            print("   💡 This could cause bias toward majority classes")
            
            # Find majority class
            majority_class = max(distribution, key=distribution.get)
            print(f"   📈 Majority class: {majority_class} ({distribution[majority_class]} images)")

def main():
    """Main function to run all diagnostics"""
    print("🔧 CNN MODEL DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Analyze training data first
    analyze_training_data_distribution()
    
    # Load and test model
    model, classes = load_model()
    
    if model is not None:
        test_with_sample_images(model, classes)
    else:
        print("❌ Cannot run tests without model")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("   1. Check if model is overfitting to handlooms category")
    print("   2. Verify data preprocessing matches training exactly")
    print("   3. Consider retraining with balanced dataset")
    print("   4. Test with completely new images outside training set")

if __name__ == "__main__":
    main()