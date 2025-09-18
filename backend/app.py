from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from PIL import Image
import os
import uuid
import base64
from datetime import datetime
import random
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'artisans-hub-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///artisans_hub.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if os.environ.get('FLASK_ENV') == 'development':
    allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
else:
    cors_origins_env = os.environ.get('CORS_ORIGINS')
    allowed_origins = cors_origins_env.split(',') if cors_origins_env else ['*']

CORS(app, origins=allowed_origins)

db = SQLAlchemy(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class Seller(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    mobile = db.Column(db.String(20), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    products = db.relationship('Product', backref='seller', lazy=True)

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'mobile': self.mobile, 'location': self.location, 'created_at': self.created_at.isoformat()}

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    seller_id = db.Column(db.Integer, db.ForeignKey('seller.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float, nullable=False)
    image_filename = db.Column(db.String(200), nullable=False)
    ai_generated = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        base_url = request.url_root.replace('http://', 'https://') if 'DYNO' in os.environ else request.url_root
        image_url = f"{base_url}uploads/{self.image_filename}"
        return {
            'id': self.id, 
            'seller_id': self.seller_id, 
            'category': self.category, 
            'description': self.description, 
            'price': self.price, 
            'image_url': image_url, 
            'ai_generated': self.ai_generated, 
            'created_at': self.created_at.isoformat(), 
            'seller': self.seller.to_dict() if self.seller else None
        }

trained_model = None
model_classes = []

def load_trained_cnn_model():
    global trained_model, model_classes
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        
        class HandicraftCNN(nn.Module):
            def __init__(self, num_classes=4):
                super(HandicraftCNN, self).__init__()
                self.backbone = models.resnet18(weights=None)
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(0.5), 
                    nn.Linear(self.backbone.fc.in_features, 128), 
                    nn.ReLU(), 
                    nn.Dropout(0.3), 
                    nn.Linear(128, num_classes)
                )
            def forward(self, x):
                return self.backbone(x)
        
        model_path = 'models/handicraft_cnn.pth'
        if os.path.exists(model_path):
            device = torch.device('cpu')
            model = HandicraftCNN(num_classes=4)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            trained_model = model
            model_classes = checkpoint.get('classes', ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls'])
            print(f"✅ CNN model loaded successfully! Accuracy: {checkpoint.get('accuracy', 'N/A')}%")
            return True
        else:
            print("⚠️ CNN model file not found. AI analysis will be disabled.")
            return False
    except Exception as e:
        print(f"❌ Failed to load CNN model: {e}")
        return False

def cnn_image_analysis(image_path):
    global trained_model, model_classes
    if not trained_model:
        raise Exception("CNN model is not loaded.")
    try:
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = trained_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        category_mapping = {
            'basket_weaving': 'Basket Weaving', 
            'handlooms': 'Handlooms', 
            'pottery': 'Pottery', 
            'wooden_dolls': 'Wooden Dolls'
        }
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class_name = model_classes[predicted_idx]
        
        return {
            'predicted_category': category_mapping.get(predicted_class_name, "Unknown"),
            'confidence': round(probabilities[predicted_idx].item(), 3)
        }
    except Exception as e:
        print(f"CNN prediction failed: {e}")
        raise e

def process_uploaded_image(base64_string):
    try:
        header, encoded = base64_string.split(",", 1)
        image_data = base64.b64decode(encoded)
        filename = f"{uuid.uuid4().hex}.jpeg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(image_path, "wb") as f:
            f.write(image_data)
            
        return filename
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

def generate_ai_description(category):
    descriptions = {
        "Pottery": "A beautifully handcrafted piece of pottery, showcasing traditional techniques. Fired to perfection, this durable item is perfect for home decor or daily use.",
        "Basket Weaving": "Intricately woven by skilled artisans, this basket is made from natural, eco-friendly materials. It's both a practical storage solution and a rustic decorative piece.",
        "Handlooms": "This vibrant handloom textile is a testament to timeless weaving traditions. Made with high-quality thread, its rich colors and patterns will brighten any space.",
        "Wooden Dolls": "A charming, hand-carved wooden doll, painted with non-toxic colors. This unique toy reflects cultural heritage and makes for a wonderful collectible or gift.",
        "Unknown": "A unique piece of artisan craft. Its quality and design speak for themselves, making it a valuable addition to any collection."
    }
    return descriptions.get(category, descriptions["Unknown"])

def generate_ai_price(category):
    base_prices = {
        "Pottery": 800, "Basket Weaving": 1200, "Handlooms": 2500, "Wooden Dolls": 600, "Unknown": 500
    }
    base = base_prices.get(category, 500)
    suggested = base + random.randint(-150, 150)
    return {
        'suggested_price': max(100, suggested),
        'min_price': max(50, int(suggested * 0.8)),
        'max_price': int(suggested * 1.2)
    }

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/upload-analyze', methods=['POST'])
def upload_and_analyze():
    try:
        data = request.get_json()
        if 'image' not in data or not data['image']:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        filename = process_uploaded_image(data['image'])
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        analysis = cnn_image_analysis(image_path)
        category = analysis['predicted_category']
        
        description = generate_ai_description(category)
        pricing = generate_ai_price(category)
        
        return jsonify({
            'success': True, 
            'analysis': analysis, 
            'ai_description': description, 
            'pricing_suggestion': pricing, 
            'image_filename': filename
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'An internal error occurred: {str(e)}'}), 500

@app.route('/api/create-listing', methods=['POST'])
def create_listing():
    data = request.get_json()
    required_fields = ['seller_name', 'seller_mobile', 'seller_location', 'category', 'description', 'price', 'image_filename']
    if not all(field in data for field in required_fields):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400

    try:
        seller = Seller.query.filter_by(mobile=data['seller_mobile']).first()
        if not seller:
            seller = Seller(name=data['seller_name'], mobile=data['seller_mobile'], location=data['seller_location'])
            db.session.add(seller)
            db.session.commit()
        
        new_product = Product(
            seller_id=seller.id,
            category=data['category'],
            description=data['description'],
            price=float(data['price']),
            image_filename=data['image_filename'],
            ai_generated=data.get('ai_generated', True)
        )
        db.session.add(new_product)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Listing created successfully', 'product_id': new_product.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500

@app.route('/api/products', methods=['GET'])
def get_all_products():
    try:
        products = Product.query.order_by(Product.created_at.desc()).all()
        return jsonify([product.to_dict() for product in products])
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

with app.app_context():
    db.create_all()
    load_trained_cnn_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)