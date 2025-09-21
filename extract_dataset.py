"""
Dataset Extraction and Organization Script
Extracts your ZIP dataset and organizes it for CNN training
"""

import zipfile
import os
import shutil
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

def extract_and_organize_dataset():
    """Extract ZIP dataset and organize for CNN training"""
    
    # Your dataset path
    dataset_zip = r"C:\Users\varma\OneDrive\Desktop\DATASET-20250816T124743Z-1-001.zip"
    
    # Project paths
    project_root = Path(r"c:\Users\varma\New folder (3)")
    data_raw_dir = project_root / "data" / "raw"
    temp_extract_dir = project_root / "temp_dataset"
    
    # CNN training categories with keywords for auto-categorization
    categories = {
        'basket_weaving': ['basket', 'wicker', 'bamboo', 'weaving', 'fiber', 'cane', 'rattan'],
        'handlooms': ['textile', 'fabric', 'cloth', 'weave', 'handloom', 'saree', 'silk', 'cotton', 'loom'],
        'pottery': ['pot', 'ceramic', 'clay', 'pottery', 'earthenware', 'vessel', 'bowl', 'jar'],
        'wooden_dolls': ['doll', 'wooden', 'figurine', 'carving', 'wood', 'toy', 'statue', 'craft']
    }
    
    print("🚀 EXTRACTING AND ORGANIZING YOUR DATASET")
    print("=" * 50)
    
    # Step 1: Check if ZIP file exists
    if not os.path.exists(dataset_zip):
        print(f"❌ Dataset ZIP not found at: {dataset_zip}")
        print("Please verify the path is correct.")
        return False
    
    print(f"✅ Found dataset ZIP: {os.path.basename(dataset_zip)}")
    file_size = os.path.getsize(dataset_zip) / (1024 * 1024)  # Size in MB
    print(f"📦 File size: {file_size:.1f} MB")
    
    # Step 2: Create directories
    os.makedirs(temp_extract_dir, exist_ok=True)
    for category in categories.keys():
        os.makedirs(data_raw_dir / category, exist_ok=True)
    
    print("✅ Created training directories")
    
    # Step 3: Extract ZIP file
    try:
        print("📦 Extracting ZIP file...")
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        print("✅ ZIP file extracted successfully")
    except Exception as e:
        print(f"❌ Error extracting ZIP: {e}")
        return False
    
    # Step 4: Find and organize images
    print("🔍 Scanning extracted files...")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    found_images = []
    
    # Recursively find all images
    for root, dirs, files in os.walk(temp_extract_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                found_images.append(os.path.join(root, file))
    
    print(f"✅ Found {len(found_images)} images total")
    
    if len(found_images) == 0:
        print("❌ No images found in the ZIP file!")
        shutil.rmtree(temp_extract_dir)
        return False
    
    # Step 5: Categorize images based on filename/folder/content
    categorized_images = {cat: [] for cat in categories.keys()}
    uncategorized_images = []
    
    print("🏷️ Categorizing images...")
    
    for img_path in found_images:
        img_name = os.path.basename(img_path).lower()
        img_folder = os.path.dirname(img_path).lower()
        full_path = img_path.lower()
        
        categorized = False
        
        # Check filename, folder, and full path for category keywords
        for category, keywords in categories.items():
            if any(keyword in img_name or keyword in img_folder or keyword in full_path for keyword in keywords):
                categorized_images[category].append(img_path)
                categorized = True
                break
        
        if not categorized:
            uncategorized_images.append(img_path)
    
    # Step 6: Copy images to training folders with validation
    print("\n📂 ORGANIZING IMAGES BY CATEGORY:")
    total_organized = 0
    
    for category, images in categorized_images.items():
        valid_images = 0
        
        if images:
            print(f"   Processing {category}: {len(images)} images found...")
            
            for i, img_path in enumerate(images):
                try:
                    # Validate image using OpenCV (following best practices from memory)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Additional validation with PIL
                        pil_img = Image.open(img_path)
                        pil_img.verify()  # Verify image integrity
                        
                        # Copy to training folder with sequential naming
                        dest_filename = f"{category}_{valid_images+1:03d}{Path(img_path).suffix.lower()}"
                        dest_path = data_raw_dir / category / dest_filename
                        shutil.copy2(img_path, dest_path)
                        valid_images += 1
                        total_organized += 1
                    else:
                        print(f"      ⚠️ Skipped invalid image: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"      ⚠️ Error processing {os.path.basename(img_path)}: {str(e)[:50]}...")
        
        # Final count for this category
        status = "✅ Ready" if valid_images >= 20 else "⚠️ Need more" if valid_images > 0 else "❌ Empty"
        print(f"   {category}: {valid_images} valid images - {status}")
    
    # Step 7: Handle uncategorized images
    if uncategorized_images:
        print(f"\n⚠️ UNCATEGORIZED IMAGES: {len(uncategorized_images)}")
        print("These images couldn't be automatically categorized:")
        
        # Create uncategorized folder for manual review
        uncategorized_dir = project_root / "uncategorized_images"
        os.makedirs(uncategorized_dir, exist_ok=True)
        
        # Show and copy first 20 uncategorized images for manual review
        manual_count = 0
        for img in uncategorized_images[:20]:
            try:
                # Validate before copying
                test_img = cv2.imread(img)
                if test_img is not None:
                    dest_name = f"uncategorized_{manual_count+1:03d}{Path(img).suffix}"
                    shutil.copy2(img, uncategorized_dir / dest_name)
                    manual_count += 1
                    print(f"   • {os.path.basename(img)} → {dest_name}")
            except:
                pass
        
        if len(uncategorized_images) > 20:
            print(f"   ... and {len(uncategorized_images) - 20} more (see uncategorized_images/ folder)")
        
        print(f"\n💡 MANUAL CATEGORIZATION:")
        print(f"   📁 Check: uncategorized_images/ folder")
        print("   🔄 Move images to correct categories:")
        for category in categories.keys():
            print(f"      📂 data/raw/{category}/")
    
    # Step 8: Clean up temporary files
    print(f"\n🧹 Cleaning up temporary files...")
    shutil.rmtree(temp_extract_dir)
    
    # Step 9: Summary and training readiness check
    print(f"\n✅ DATASET ORGANIZATION COMPLETE!")
    print("=" * 50)
    print(f"📊 Total images organized: {total_organized}")
    
    # Check training readiness
    training_ready = True
    min_images_needed = 0
    
    for category in categories.keys():
        # Count all image types
        category_path = data_raw_dir / category
        count = len([f for f in category_path.glob("*") if f.suffix.lower() in image_extensions])
        
        if count < 20:
            training_ready = False
            min_images_needed += (20 - count)
    
    if training_ready:
        print(f"\n🎉 READY FOR CNN TRAINING!")
        print("Your dataset meets the minimum requirements!")
        print("\n🚀 NEXT STEPS:")
        print("1. Run CNN training:")
        print("   python ai_models/train_classifier.py")
        print("2. This will train with transfer learning and data augmentation")
        print("3. Expected accuracy: 80-95% with your dataset")
    else:
        print(f"\n📝 TRAINING READINESS:")
        print(f"   ⚠️ Need {min_images_needed} more images total")
        print("   📋 Recommendations:")
        print("   • Add more images to categories with <20 images")
        print("   • Check uncategorized_images/ folder for manual sorting")
        print("   • Aim for 50+ images per category for best results")
        print("\n🔄 STILL TRAINABLE:")
        print("   • Can train with current data using data augmentation")
        print("   • Synthetic data generation will create more training samples")
    
    return True

def show_dataset_stats():
    """Show current dataset statistics"""
    project_root = Path(r"c:\Users\varma\New folder (3)")
    data_raw_dir = project_root / "data" / "raw"
    
    print("\n📊 CURRENT DATASET STATISTICS:")
    print("-" * 30)
    
    categories = ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls']
    total_images = 0
    
    for category in categories:
        category_path = data_raw_dir / category
        if category_path.exists():
            count = len(list(category_path.glob("*.jpg"))) + \
                   len(list(category_path.glob("*.png"))) + \
                   len(list(category_path.glob("*.jpeg")))
            total_images += count
            status = "✅" if count >= 20 else "⚠️" if count > 0 else "❌"
            print(f"   {status} {category}: {count} images")
        else:
            print(f"   ❌ {category}: 0 images (folder missing)")
    
    print(f"\n   📈 Total: {total_images} images")
    
    if total_images >= 80:
        print("   🎯 Status: Ready for training!")
    elif total_images >= 40:
        print("   🔄 Status: Can train with augmentation")
    else:
        print("   📝 Status: Need more images for optimal training")

if __name__ == "__main__":
    # Show current stats first
    show_dataset_stats()
    
    print("\n" + "="*60)
    
    # Extract and organize the new dataset
    success = extract_and_organize_dataset()
    
    if success:
        print("\n" + "="*60)
        # Show updated stats
        show_dataset_stats()
    else:
        print("\n❌ Dataset extraction failed. Please check the ZIP file path.")