
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import random
from flask import current_app

# Globals for holding the loaded model and class mapping
trained_model = None
idx_to_class = None

# This class is a placeholder for the architecture, the actual model is loaded below.
class HandicraftCNN(nn.Module):
    def __init__(self, num_classes):
        super(HandicraftCNN, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

def load_trained_cnn_model():
    """Loads the trained model and class indices from disk into memory."""
    global trained_model, idx_to_class

    model_dir = os.path.join(current_app.root_path, '../models')
    model_path = os.path.join(model_dir, 'handicraft_cnn.pth')
    indices_path = os.path.join(model_dir, 'class_indices.json')

    if not all(os.path.exists(p) for p in [model_path, indices_path]):
        print(f"⚠️  Warning: Model or class indices not found in '{model_dir}'. AI features will be disabled.")
        return

    try:
        with open(indices_path, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # FIX: Define the model architecture EXACTLY as it was during training
        model = models.resnet18(weights=None) # Start with an untrained ResNet-18
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) # Replace the final layer
        
        # Now, load the state dictionary into this matching structure
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        trained_model = model
        print(f"✅ CNN model and class indices loaded successfully.")

    except Exception as e:
        print(f"❌ An error occurred while loading the AI model: {e}")

def cnn_image_analysis(image_path):
    """Analyzes an image and returns the predicted category and confidence."""
    if not trained_model or not idx_to_class:
        raise Exception("CNN model or class indices are not loaded.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = trained_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx_tensor = torch.max(probabilities, 0)
        predicted_idx = predicted_idx_tensor.item()

    predicted_class_name = idx_to_class.get(predicted_idx, "Unknown")
    
    return {
        'predicted_category': predicted_class_name.replace('_', ' ').title(),
        'confidence': round(confidence.item(), 3)
    }

# --- Placeholder Functions ---
def generate_ai_description(category):
    """Generates a placeholder description based on the category."""
    descriptions = {
        'pottery': "A beautifully crafted piece of pottery, showcasing traditional techniques.",
        'wooden_dolls': "An exquisite hand-carved wooden doll, rich in cultural heritage.",
        'basket_weaving': "A skillfully woven basket, made from natural, sustainable materials.",
        'handlooms': "A vibrant handloom textile, woven with intricate patterns and colors."
    }
    return descriptions.get(category.lower().replace(' ', '_'), "A unique, handcrafted item.")

def generate_ai_price(category):
    """Generates a placeholder price based on the category."""
    price_ranges = {
        'pottery': (500, 3000),
        'wooden_dolls': (300, 1500),
        'basket_weaving': (250, 1200),
        'handlooms': (1000, 8000)
    }
    min_price, max_price = price_ranges.get(category.lower().replace(' ', '_'), (200, 1000))
    return {
        'suggested_price': random.randint(min_price, max_price),
        'min_price': min_price,
        'max_price': max_price
    }
