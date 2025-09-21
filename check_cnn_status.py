"""
CNN Model Setup and Training Data Helper
This script helps diagnose and fix CNN model issues
"""

import os
import cv2
import numpy as np
from pathlib import Path

def check_data_status():
    """Check current training data status"""
    print("🔍 CHECKING CNN MODEL STATUS")
    print("=" * 50)
    
    data_dir = Path("data/raw")
    categories = ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls']
    
    print("\n📊 Training Data Status:")
    total_images = 0
    
    for category in categories:
        category_path = data_dir / category
        if category_path.exists():
            image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png")) + list(category_path.glob("*.jpeg"))
            count = len(image_files)
            total_images += count
            
            status = "✅ Good" if count >= 20 else "⚠️ Too Few" if count > 0 else "❌ Empty"
            print(f"   {category}: {count} images - {status}")
        else:
            print(f"   {category}: Directory missing - ❌ Not Found")
    
    print(f"\n📈 Total Images: {total_images}")
    
    if total_images == 0:
        print("\n❌ PROBLEM: No training data found!")
        print("💡 SOLUTION: Add images to the category folders")
        print_data_requirements()
    elif total_images < 80:  # 20 per category minimum
        print("\n⚠️ PROBLEM: Insufficient training data!")
        print("💡 SOLUTION: Add more images for better accuracy")
        print_data_requirements()
    else:
        print("\n✅ GOOD: Sufficient data available for training!")
        return True
    
    return False

def print_data_requirements():
    """Print requirements for training data"""
    print("\n📝 TRAINING DATA REQUIREMENTS:")
    print("   • Minimum: 20 images per category (80 total)")
    print("   • Recommended: 50+ images per category (200+ total)")
    print("   • Optimal: 100+ images per category (400+ total)")
    
    print("\n📁 ADD IMAGES TO THESE FOLDERS:")
    categories = {
        'basket_weaving': 'Wicker baskets, bamboo baskets, fiber crafts',
        'handlooms': 'Textiles, fabrics, woven materials, sarees',
        'pottery': 'Clay pots, ceramic items, earthenware',
        'wooden_dolls': 'Wooden figurines, carved dolls, wooden toys'
    }
    
    for category, description in categories.items():
        print(f"   📂 data/raw/{category}/ → {description}")
    
    print("\n🖼️ IMAGE REQUIREMENTS:")
    print("   • Format: JPG, PNG, JPEG")
    print("   • Quality: Clear, well-lit images")
    print("   • Variety: Different angles, backgrounds, lighting")
    print("   • Size: Any size (will be resized automatically)")

def check_model_integration():
    """Check if CNN model is properly integrated in backend"""
    print("\n🔧 CHECKING MODEL INTEGRATION:")
    
    backend_file = Path("backend/app.py")
    if backend_file.exists():
        with open(backend_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if actual CNN model is used
        cnn_imports = [
            "from ai_models.cnn_classifier.enhanced_classifier import",
            "EnhancedHandicraftClassifier",
            "torch"
        ]
        
        cnn_usage = any(imp in content for imp in cnn_imports)
        
        if cnn_usage:
            print("   ✅ CNN model imports found")
        else:
            print("   ❌ CNN model not imported in backend")
            
        # Check if simple heuristics are being used instead
        if "simple_image_analysis" in content:
            print("   ⚠️ Currently using simple color heuristics")
            print("   💡 Need to replace with actual CNN model")
        
        return cnn_usage
    else:
        print("   ❌ Backend file not found")
        return False

def run_training_check():
    """Check if training can be run"""
    print("\n🏃 TRAINING READINESS CHECK:")
    
    # Check if training script exists
    training_script = Path("ai_models/train_classifier.py")
    if training_script.exists():
        print("   ✅ Training script found")
    else:
        print("   ❌ Training script missing")
        return False
    
    # Check dependencies
    try:
        import torch
        print("   ✅ PyTorch available")
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False
    
    try:
        import torchvision
        print("   ✅ Torchvision available") 
    except ImportError:
        print("   ❌ Torchvision not installed")
        return False
    
    return True

def suggest_next_steps():
    """Suggest next steps based on current status"""
    print("\n🚀 NEXT STEPS TO FIX CNN MODEL:")
    print("=" * 40)
    
    data_ready = check_data_status()
    model_integrated = check_model_integration() 
    training_ready = run_training_check()
    
    print("\n📋 ACTION PLAN:")
    
    if not data_ready:
        print("   1. ⭐ ADD TRAINING DATA (Critical)")
        print("      → Collect 20+ images per category")
        print("      → Save in data/raw/[category_name]/ folders")
        print("      → Run this script again to verify")
    
    if not training_ready:
        print("   2. ⭐ INSTALL DEPENDENCIES")
        print("      → pip install torch torchvision")
        print("      → pip install opencv-python pillow")
    
    if data_ready and training_ready:
        print("   3. ⭐ TRAIN THE MODEL")
        print("      → python ai_models/train_classifier.py")
        print("      → Wait for training to complete")
    
    if not model_integrated:
        print("   4. ⭐ INTEGRATE TRAINED MODEL")
        print("      → Update backend to use trained CNN")
        print("      → Replace simple heuristics with CNN predictions")
    
    print("\n🎯 EXPECTED RESULTS AFTER FIXES:")
    print("   • 80-95% classification accuracy")
    print("   • Proper image recognition")
    print("   • AI-powered category detection")
    print("   • Better user experience")

if __name__ == "__main__":
    suggest_next_steps()