"""
Synthetic Dataset Generation for Handicraft Classification
Advanced techniques to create realistic synthetic data for small datasets
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
import os
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, rotate
import json
from datetime import datetime


class StyleTransferGenerator:
    """
    Generate synthetic images using neural style transfer techniques
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def apply_artistic_style(self, content_image: np.ndarray, 
                           style_type: str = 'pottery') -> np.ndarray:
        """Apply artistic styles to create variations"""
        
        # Convert to PIL for easier manipulation
        img = Image.fromarray(content_image)
        
        # Style-specific transformations
        if style_type == 'pottery':
            return self._pottery_style(img)
        elif style_type == 'wooden_dolls':
            return self._wood_style(img)
        elif style_type == 'basket_weaving':
            return self._weaving_style(img)
        elif style_type == 'handlooms':
            return self._textile_style(img)
        
        return np.array(img)
    
    def _pottery_style(self, img: Image.Image) -> np.ndarray:
        """Apply pottery-specific artistic transformations"""
        # Enhance earthy colors
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.8)  # Reduce saturation
        
        # Add clay-like texture
        img = img.filter(ImageFilter.EMBOSS)
        
        # Warm color shift
        arr = np.array(img)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.1, 0, 255)  # Enhance red
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.95, 0, 255)  # Reduce green
        
        return arr.astype(np.uint8)
    
    def _wood_style(self, img: Image.Image) -> np.ndarray:
        """Apply wood-specific transformations"""
        # Enhance brown tones
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)
        
        # Add wood grain effect
        arr = np.array(img)
        
        # Create wood grain pattern
        grain = np.random.normal(0, 5, arr.shape[:2])
        for i in range(3):
            arr[:, :, i] = np.clip(arr[:, :, i] + grain, 0, 255)
        
        return arr.astype(np.uint8)
    
    def _weaving_style(self, img: Image.Image) -> np.ndarray:
        """Apply basket weaving patterns"""
        arr = np.array(img)
        h, w = arr.shape[:2]
        
        # Create weaving pattern overlay
        pattern = np.zeros((h, w))
        
        # Horizontal lines
        for i in range(0, h, 8):
            pattern[i:i+3, :] = 0.1
        
        # Vertical lines
        for j in range(0, w, 8):
            pattern[:, j:j+3] = 0.1
        
        # Apply pattern
        for i in range(3):
            arr[:, :, i] = np.clip(arr[:, :, i] * (1 - pattern), 0, 255)
        
        return arr.astype(np.uint8)
    
    def _textile_style(self, img: Image.Image) -> np.ndarray:
        """Apply textile/handloom patterns"""
        arr = np.array(img)
        
        # Add fabric texture
        noise = np.random.normal(0, 3, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        
        # Enhance texture
        enhancer = ImageEnhance.Sharpness(Image.fromarray(arr.astype(np.uint8)))
        img_enhanced = enhancer.enhance(1.5)
        
        return np.array(img_enhanced)


class GeometricAugmentationGenerator:
    """
    Generate synthetic data using geometric transformations and compositions
    """
    
    def __init__(self):
        self.background_colors = [
            (240, 235, 220),  # Cream
            (245, 245, 220),  # Beige
            (255, 248, 220),  # Cornsilk
            (250, 240, 230),  # Linen
            (245, 222, 179),  # Wheat
        ]
    
    def create_composite_image(self, base_images: List[np.ndarray], 
                             class_id: int) -> np.ndarray:
        """Create composite images by combining multiple objects"""
        
        # Select 2-3 random base images
        selected = random.sample(base_images, min(3, len(base_images)))
        
        # Create canvas
        canvas_size = (224, 224, 3)
        canvas = np.full(canvas_size, random.choice(self.background_colors), dtype=np.uint8)
        
        # Place objects on canvas
        for i, img in enumerate(selected):
            # Resize object
            scale = random.uniform(0.3, 0.8)
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            resized = cv2.resize(img, new_size)
            
            # Random position
            max_x = canvas_size[1] - resized.shape[1]
            max_y = canvas_size[0] - resized.shape[0]
            
            if max_x > 0 and max_y > 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                # Blend with canvas
                alpha = random.uniform(0.7, 1.0)
                # Convert to float32 for safe arithmetic operations
                resized_float = resized.astype(np.float32)
                canvas_section = canvas[y:y+resized.shape[0], x:x+resized.shape[1]].astype(np.float32)
                blended = alpha * resized_float + (1-alpha) * canvas_section
                canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return canvas.astype(np.uint8)
    
    def generate_geometric_variations(self, image: np.ndarray, 
                                   num_variations: int = 10) -> List[np.ndarray]:
        """Generate multiple geometric variations of an image"""
        variations = []
        
        for _ in range(num_variations):
            # Start with original
            var_img = image.copy()
            
            # Apply random transformations
            transformations = [
                self._apply_perspective_transform,
                self._apply_shear_transform,
                self._apply_elastic_deformation,
                self._apply_barrel_distortion,
            ]
            
            # Apply 1-2 random transformations
            selected_transforms = random.sample(transformations, random.randint(1, 2))
            
            for transform in selected_transforms:
                try:
                    var_img = transform(var_img)
                except:
                    continue  # Skip if transformation fails
            
            variations.append(var_img)
        
        return variations
    
    def _apply_perspective_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply random perspective transformation"""
        h, w = img.shape[:2]
        
        # Define source points (corners)
        src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # Random destination points for perspective effect
        offset = min(w, h) * 0.1
        dst_points = np.array([
            [random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
            [random.uniform(-offset, offset), h + random.uniform(-offset, offset)]
        ], dtype=np.float32)
        
        # Apply perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(img, matrix, (w, h))
        
        return transformed
    
    def _apply_shear_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply random shear transformation"""
        h, w = img.shape[:2]
        
        # Shear parameters
        shear_x = random.uniform(-0.2, 0.2)
        shear_y = random.uniform(-0.2, 0.2)
        
        # Create transformation matrix
        M = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
        
        # Apply transformation
        sheared = cv2.warpAffine(img, M, (w, h))
        
        return sheared
    
    def _apply_elastic_deformation(self, img: np.ndarray) -> np.ndarray:
        """Apply elastic deformation"""
        h, w = img.shape[:2]
        
        # Create displacement fields
        dx = gaussian_filter(np.random.random((h, w)) * 2 - 1, sigma=5) * 5
        dy = gaussian_filter(np.random.random((h, w)) * 2 - 1, sigma=5) * 5
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w - 1)
        y_new = np.clip(y + dy, 0, h - 1)
        
        # Apply deformation
        deformed = cv2.remap(img, x_new.astype(np.float32), y_new.astype(np.float32), 
                           cv2.INTER_LINEAR)
        
        return deformed
    
    def _apply_barrel_distortion(self, img: np.ndarray) -> np.ndarray:
        """Apply barrel/pincushion distortion"""
        h, w = img.shape[:2]
        
        # Distortion parameters
        k1 = random.uniform(-0.0005, 0.0005)
        k2 = random.uniform(-0.00001, 0.00001)
        
        # Camera matrix (simplified)
        cx, cy = w / 2, h / 2
        fx = fy = max(w, h)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize coordinates
        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fy
        
        # Calculate radial distance
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # Apply distortion
        distortion = 1 + k1 * r**2 + k2 * r**4
        x_distorted = x_norm * distortion * fx + cx
        y_distorted = y_norm * distortion * fy + cy
        
        # Remap image
        distorted = cv2.remap(img, x_distorted.astype(np.float32), 
                            y_distorted.astype(np.float32), cv2.INTER_LINEAR)
        
        return distorted


class ColorSpaceGenerator:
    """
    Generate synthetic data using color space manipulations
    """
    
    def __init__(self):
        # Define color palettes for different handicraft types
        self.color_palettes = {
            'pottery': {
                'primary': [(139, 69, 19), (160, 82, 45), (210, 180, 140)],  # Browns
                'secondary': [(205, 133, 63), (222, 184, 135), (245, 245, 220)]
            },
            'wooden_dolls': {
                'primary': [(101, 67, 33), (139, 90, 43), (160, 120, 90)],  # Wood tones
                'secondary': [(205, 175, 149), (222, 196, 176), (245, 235, 220)]
            },
            'basket_weaving': {
                'primary': [(218, 165, 32), (184, 134, 11), (255, 215, 0)],  # Straw colors
                'secondary': [(240, 230, 140), (250, 240, 230), (255, 248, 220)]
            },
            'handlooms': {
                'primary': [(220, 20, 60), (0, 100, 0), (0, 0, 139)],  # Vibrant colors
                'secondary': [(255, 182, 193), (144, 238, 144), (173, 216, 230)]
            }
        }
    
    def generate_color_variations(self, image: np.ndarray, class_name: str, 
                                num_variations: int = 15) -> List[np.ndarray]:
        """Generate color variations specific to handicraft type"""
        variations = []
        palette = self.color_palettes.get(class_name, self.color_palettes['pottery'])
        
        for _ in range(num_variations):
            var_img = image.copy()
            
            # Apply color transformations
            transformations = [
                lambda img: self._shift_to_palette(img, palette['primary']),
                lambda img: self._shift_to_palette(img, palette['secondary']),
                lambda img: self._apply_color_temperature(img),
                lambda img: self._apply_vintage_effect(img),
                lambda img: self._apply_material_specific_colors(img, class_name),
            ]
            
            # Apply random transformation
            transform = random.choice(transformations)
            var_img = transform(var_img)
            
            variations.append(var_img)
        
        return variations
    
    def _shift_to_palette(self, img: np.ndarray, palette: List[Tuple[int, int, int]]) -> np.ndarray:
        """Shift image colors to match a specific palette"""
        # Convert to Lab color space for better color manipulation
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Extract dominant colors using K-means
        pixels = lab.reshape(-1, 3)
        kmeans = KMeans(n_clusters=min(len(palette), 3), random_state=42)
        kmeans.fit(pixels)
        
        # Map to palette colors
        palette_lab = []
        for color in palette:
            color_array = np.array([[[color[0], color[1], color[2]]]], dtype=np.uint8)
            lab_color = cv2.cvtColor(color_array, cv2.COLOR_RGB2LAB)[0][0]
            palette_lab.append(lab_color)
        
        # Replace cluster centers with palette colors
        new_img = lab.copy()
        if hasattr(kmeans, 'labels_') and kmeans.labels_ is not None:
            for i, center in enumerate(kmeans.cluster_centers_):
                if i < len(palette_lab):
                    mask = kmeans.labels_.reshape(lab.shape[:2]) == i
                    new_img[mask] = palette_lab[i]
        
        # Convert back to RGB
        result = cv2.cvtColor(new_img, cv2.COLOR_LAB2RGB)
        return result
    
    def _apply_color_temperature(self, img: np.ndarray) -> np.ndarray:
        """Apply random color temperature adjustment"""
        # Random temperature shift
        temperature = random.uniform(0.8, 1.2)
        
        # Apply to red and blue channels
        result = img.copy().astype(np.float32)
        result[:, :, 0] *= temperature  # Red channel
        result[:, :, 2] *= (2 - temperature)  # Blue channel (inverse)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_vintage_effect(self, img: np.ndarray) -> np.ndarray:
        """Apply vintage/aged effect"""
        try:
            # Ensure img is in correct format
            if img is None or img.size == 0:
                raise ValueError("Invalid input image")
            
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Check if image was loaded successfully before applying cv2.cvtColor
            if img.shape[2] != 3:
                raise ValueError("Image must have 3 channels (RGB)")
            
            # Reduce saturation
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            if hsv is None:
                raise ValueError("Failed to convert to HSV")
            
            # Apply saturation reduction with safe type conversion
            saturation_factor = random.uniform(0.6, 0.9)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
            
            # Add warm tint
            aged = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            if aged is None:
                raise ValueError("Failed to convert back to RGB")
            
            # Enhance colors with safe arithmetic
            aged = aged.astype(np.float32)
            aged[:, :, 0] = np.clip(aged[:, :, 0] * 1.1, 0, 255)  # Enhance red
            aged[:, :, 1] = np.clip(aged[:, :, 1] * 1.05, 0, 255)  # Slight green
            
            return aged.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in vintage effect: {e}")
            # Return original image if processing fails
            return img.astype(np.uint8) if img.dtype != np.uint8 else img
    
    def _apply_material_specific_colors(self, img: np.ndarray, class_name: str) -> np.ndarray:
        """Apply colors specific to material type"""
        try:
            # Ensure img is in correct format
            if img is None or img.size == 0:
                raise ValueError("Invalid input image")
            
            # Convert to float32 for calculations to avoid overflow
            img_float = img.astype(np.float32)
            
            if class_name == 'pottery':
                # Enhance earth tones
                img_float[:, :, 0] = np.clip(img_float[:, :, 0] * 1.2, 0, 255)  # Red
                img_float[:, :, 1] = np.clip(img_float[:, :, 1] * 0.9, 0, 255)   # Green
                img_float[:, :, 2] = np.clip(img_float[:, :, 2] * 0.8, 0, 255)   # Blue
            elif class_name == 'wooden_dolls':
                # Enhance brown/wood tones
                img_float[:, :, 0] = np.clip(img_float[:, :, 0] * 1.1, 0, 255)
                img_float[:, :, 1] = np.clip(img_float[:, :, 1] * 0.95, 0, 255)
            elif class_name == 'handlooms':
                # Enhance vibrant textile colors
                img_float[:, :, 0] = np.clip(img_float[:, :, 0] * 1.3, 0, 255)
                img_float[:, :, 2] = np.clip(img_float[:, :, 2] * 1.2, 0, 255)
            
            return img_float.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in material-specific colors: {e}")
            # Return original image if processing fails
            return img.astype(np.uint8) if img.dtype != np.uint8 else img


class SyntheticDatasetGenerator:
    """
    Main class to orchestrate all synthetic data generation techniques
    """
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.style_generator = StyleTransferGenerator()
        self.geometric_generator = GeometricAugmentationGenerator()
        self.color_generator = ColorSpaceGenerator()
        
    def generate_synthetic_dataset(self, original_images: Dict[str, List[np.ndarray]], 
                                 target_size_per_class: int = 500) -> Dict[str, List[np.ndarray]]:
        """
        Generate comprehensive synthetic dataset
        
        Args:
            original_images: Dict mapping class names to lists of images
            target_size_per_class: Target number of images per class
            
        Returns:
            Dict mapping class names to synthetic images
        """
        synthetic_data = {}
        
        print(f"🎨 Generating synthetic dataset...")
        print(f"🎯 Target size per class: {target_size_per_class}")
        
        for class_name in self.class_names:
            print(f"\n📸 Generating for class: {class_name}")
            
            if class_name not in original_images:
                print(f"⚠️  No original images found for {class_name}")
                continue
                
            class_images = original_images[class_name]
            original_count = len(class_images)
            needed = target_size_per_class - original_count
            
            if needed <= 0:
                synthetic_data[class_name] = class_images
                continue
            
            synthetic_images = class_images.copy()  # Start with originals
            
            # Calculate how many to generate with each technique
            per_technique = needed // 4
            
            # 1. Style transfer variations
            print(f"   🎭 Generating style variations: {per_technique}")
            for img in class_images:
                style_vars = [self.style_generator.apply_artistic_style(img, class_name) 
                            for _ in range(per_technique // original_count + 1)]
                synthetic_images.extend(style_vars[:per_technique // original_count])
            
            # 2. Geometric variations
            print(f"   📐 Generating geometric variations: {per_technique}")
            for img in class_images[:min(5, len(class_images))]:  # Limit to avoid too many
                geom_vars = self.geometric_generator.generate_geometric_variations(
                    img, per_technique // min(5, len(class_images)) + 1
                )
                synthetic_images.extend(geom_vars[:per_technique // min(5, len(class_images))])
            
            # 3. Color variations
            print(f"   🎨 Generating color variations: {per_technique}")
            for img in class_images:
                color_vars = self.color_generator.generate_color_variations(
                    img, class_name, per_technique // original_count + 1
                )
                synthetic_images.extend(color_vars[:per_technique // original_count])
            
            # 4. Composite images
            print(f"   🖼️  Generating composite images: {per_technique}")
            for _ in range(per_technique):
                composite = self.geometric_generator.create_composite_image(class_images, 0)
                synthetic_images.append(composite)
            
            # Trim to target size
            synthetic_data[class_name] = synthetic_images[:target_size_per_class]
            
            print(f"   ✅ Generated {len(synthetic_data[class_name])} total images "
                  f"({len(synthetic_data[class_name]) - original_count} synthetic)")
        
        return synthetic_data
    
    def save_synthetic_dataset(self, synthetic_data: Dict[str, List[np.ndarray]], 
                             output_dir: str):
        """Save synthetic dataset to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        for class_name, images in synthetic_data.items():
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i, img in enumerate(images):
                filename = f"{class_name}_synthetic_{i:04d}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'classes': list(synthetic_data.keys()),
            'images_per_class': {name: len(images) for name, images in synthetic_data.items()},
            'total_images': sum(len(images) for images in synthetic_data.values()),
            'techniques_used': [
                'Style Transfer (artistic effects)',
                'Geometric Transformations (perspective, shear, elastic)',
                'Color Space Manipulations (palette shifts, temperature)',
                'Composite Generation (multi-object scenes)'
            ]
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def visualize_generation_process(self, original_image: np.ndarray, class_name: str):
        """Visualize the synthetic generation process"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        # Original
        axes[0].imshow(original_image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Style transfer
        style_result = self.style_generator.apply_artistic_style(original_image, class_name)
        axes[1].imshow(style_result)
        axes[1].set_title('Style Transfer')
        axes[1].axis('off')
        
        # Geometric transformations
        geom_results = self.geometric_generator.generate_geometric_variations(original_image, 3)
        for i, result in enumerate(geom_results[:3]):
            axes[2 + i].imshow(result)
            axes[2 + i].set_title(f'Geometric #{i+1}')
            axes[2 + i].axis('off')
        
        # Color variations
        color_results = self.color_generator.generate_color_variations(original_image, class_name, 3)
        for i, result in enumerate(color_results[:3]):
            axes[5 + i].imshow(result)
            axes[5 + i].set_title(f'Color #{i+1}')
            axes[5 + i].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Synthetic Data Generation Process - {class_name}', y=1.02)
        plt.show()


# Example usage
if __name__ == "__main__":
    print("🎨 Synthetic Dataset Generator for Handicraft Classification")
    print("=" * 65)
    
    # Configuration
    CLASS_NAMES = ['pottery', 'wooden_dolls', 'basket_weaving', 'handlooms']
    
    # Initialize generator
    generator = SyntheticDatasetGenerator(CLASS_NAMES)
    
    print("✅ Synthetic dataset generator initialized!")
    print(f"✅ Supported classes: {CLASS_NAMES}")
    print("\n🔧 Available generation techniques:")
    print("   1. 🎭 Style Transfer - Apply artistic effects specific to each handicraft type")
    print("   2. 📐 Geometric Transformations - Perspective, shear, elastic deformations")
    print("   3. 🎨 Color Space Manipulations - Palette shifts, temperature adjustments")
    print("   4. 🖼️  Composite Generation - Multi-object scenes with realistic backgrounds")
    
    print("\n📊 Expected Results:")
    print("   - From 50 images → 500+ images per class")
    print("   - 10x dataset increase with high diversity")
    print("   - Improved model robustness and generalization")
    print("   - Realistic variations maintaining class characteristics")
    
    print("\n🚀 Usage:")
    print("   1. Prepare original images in dictionary format")
    print("   2. Call generator.generate_synthetic_dataset(original_images)")
    print("   3. Save results with generator.save_synthetic_dataset()")
    print("   4. Train your CNN with the expanded dataset!")
    
    # Demo configuration
    print(f"\n⚙️  Generator Configuration:")
    print(f"   - Style Transfer: Material-specific artistic effects")
    print(f"   - Geometric: Perspective, shear, elastic, barrel distortion")
    print(f"   - Color: Palette matching, temperature shifts, vintage effects")
    print(f"   - Composite: Multi-object scenes with realistic backgrounds")