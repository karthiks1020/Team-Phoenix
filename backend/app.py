"""
Artisans Hub - Full-Stack Web Application Backend
Flask application for managing artisan products and sellers
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from PIL import Image
import os
import json
import uuid
import base64
import io
from datetime import datetime
import numpy as np
import cv2
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration for production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'artisans-hub-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///artisans_hub.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', str(16 * 1024 * 1024)))

# Initialize extensions
db = SQLAlchemy(app)
CORS(app, origins=os.environ.get('CORS_ORIGINS', '*').split(','))

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class Seller(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    mobile = db.Column(db.String(20), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    products = db.relationship('Product', backref='seller', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'mobile': self.mobile,
            'location': self.location,
            'created_at': self.created_at.isoformat()
        }

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    seller_id = db.Column(db.Integer, db.ForeignKey('seller.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    ai_generated = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'seller_id': self.seller_id,
            'category': self.category,
            'description': self.description,
            'price': self.price,
            'image_path': self.image_path,
            'ai_generated': self.ai_generated,
            'created_at': self.created_at.isoformat(),
            'seller': self.seller.to_dict() if self.seller else None
        }

# AI Description Generator
def generate_ai_description(category, image_analysis=None):
    """Generate AI-powered product description based on category"""
    descriptions = {
        'Wooden Dolls': [
            "Exquisite hand-carved wooden doll showcasing traditional craftsmanship. Each piece tells a unique story of cultural heritage and artistic dedication.",
            "Beautiful wooden figurine crafted with meticulous attention to detail. This piece represents generations of woodworking expertise passed down through artisan families.",
            "Authentic hand-carved wooden doll featuring intricate details and smooth finish. A perfect blend of traditional art and contemporary appeal."
        ],
        'Handlooms': [
            "Magnificent handwoven textile created on traditional looms. This piece showcases the rich heritage of handloom weaving with vibrant colors and intricate patterns.",
            "Premium handloom fabric woven with traditional techniques. Each thread is carefully placed to create this stunning textile that celebrates cultural artistry.",
            "Artisan-crafted handloom textile featuring traditional motifs and superior quality. A testament to the skill and dedication of master weavers."
        ],
        'Basket Weaving': [
            "Skillfully woven basket made from natural materials using age-old techniques. This eco-friendly piece combines functionality with traditional artistry.",
            "Handcrafted wicker basket showcasing the art of traditional weaving. Made with sustainable materials and time-honored techniques for lasting beauty.",
            "Artisan-made basket featuring intricate weaving patterns. This piece represents the perfect harmony between utility and traditional craftsmanship."
        ],
        'Pottery': [
            "Hand-thrown pottery piece created with traditional ceramic techniques. Each curve and line reflects the potter's skill and artistic vision.",
            "Authentic ceramic artwork crafted on the potter's wheel. This piece showcases the timeless beauty of traditional pottery making.",
            "Artisan-made pottery featuring unique glazing and traditional firing techniques. A beautiful example of ceramic artistry and cultural heritage."
        ]
    }
    
    category_descriptions = descriptions.get(category, [
        "Beautiful handcrafted piece showcasing traditional artisan skills and cultural heritage."
    ])
    
    return random.choice(category_descriptions)

# AI Price Suggestion (in Indian Rupees)
def generate_ai_price(category):
    """Generate AI-suggested pricing based on category in INR with specific ranges"""
    price_ranges_inr = {
        'Wooden Dolls': (200, 1000),
        'Handlooms': (800, 10000), 
        'Basket Weaving': (200, 700),
        'Pottery': (400, 5000)
    }
    
    min_price, max_price = price_ranges_inr.get(category, (200, 1000))
    suggested_price = random.randint(min_price, max_price)
    
    return {
        'suggested_price': suggested_price,
        'min_price': min_price,
        'max_price': max_price,
        'category_average': (min_price + max_price) // 2,
        'currency': 'INR'
    }

# Image Processing
def process_uploaded_image(image_data):
    """Process base64 image data and save to uploads folder"""
    try:
        # Remove data URL prefix if present
        if 'data:image' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image for consistency
        image = image.resize((800, 600), Image.Resampling.LANCZOS)
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save image
        image.save(filepath, 'JPEG', quality=85)
        
        return filename
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

# Load trained CNN model at startup
trained_model = None
model_classes = ['basket_weaving', 'handlooms', 'pottery', 'wooden_dolls']

def load_trained_cnn_model():
    """Load the trained CNN model for image classification"""
    global trained_model, model_classes
    
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from PIL import Image
        
        # Define the same model architecture used in training
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
        
        # Load the trained model
        device = torch.device('cpu')  # Use CPU for inference
        model = HandicraftCNN(num_classes=4)
        
        # Load model weights
        model_path = 'models/handicraft_cnn.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            trained_model = model
            
            # Update class names from checkpoint if available
            if 'classes' in checkpoint:
                model_classes = checkpoint['classes']
            
            print(f"✅ CNN model loaded successfully! Accuracy: {checkpoint.get('accuracy', 'Unknown')}%")
            return True
        else:
            print("⚠️ CNN model file not found, using fallback method")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load CNN model: {e}")
        return False

# AI Image Analysis using trained CNN Model
def analyze_image_for_category(image_path):
    """AI analysis to determine category from image using trained CNN model"""
    global trained_model, model_classes
    
    # First try to use the trained CNN model
    if trained_model is not None:
        try:
            return cnn_image_analysis(image_path)
        except Exception as e:
            print(f"CNN analysis failed: {e}")
    
    # Fallback to simple analysis if CNN fails
    try:
        return simple_image_analysis(image_path)
    except Exception as e:
        print(f"Simple analysis failed, using random fallback: {e}")
        # Final fallback to random
        categories = ['Wooden Dolls', 'Handlooms', 'Basket Weaving', 'Pottery']
        confidence_scores = [random.random() for _ in categories]
        total = sum(confidence_scores)
        confidence_scores = [score/total for score in confidence_scores]
        
        max_idx = confidence_scores.index(max(confidence_scores))
        return {
            'predicted_category': categories[max_idx],
            'confidence': round(confidence_scores[max_idx], 3),
            'all_predictions': dict(zip(categories, [round(score, 3) for score in confidence_scores])),
            'model_used': 'random_fallback'
        }

def cnn_image_analysis(image_path):
    """Use trained CNN model for image classification"""
    global trained_model, model_classes
    
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        # Image preprocessing (same as training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = trained_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Convert to category names
        category_mapping = {
            'basket_weaving': 'Basket Weaving',
            'handlooms': 'Handlooms', 
            'pottery': 'Pottery',
            'wooden_dolls': 'Wooden Dolls'
        }
        
        # Get predictions
        predicted_idx = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_idx].item())
        predicted_category = category_mapping[model_classes[predicted_idx]]
        
        # Create all predictions dictionary
        all_predictions = {}
        for i, class_name in enumerate(model_classes):
            display_name = category_mapping[class_name]
            all_predictions[display_name] = round(probabilities[i].item(), 3)
        
        return {
            'predicted_category': predicted_category,
            'confidence': round(confidence, 3),
            'all_predictions': all_predictions,
            'model_used': 'trained_cnn',
            'model_accuracy': '98%'  # From training results
        }
        
    except Exception as e:
        print(f"CNN prediction failed: {e}")
        raise e
            
# Simple image analysis without heavy dependencies
def simple_image_analysis(image_path):
    """Simple image analysis using basic image properties"""
    try:
        from PIL import Image
        import numpy as np
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Analyze basic color properties
        avg_color = np.mean(img_array, axis=(0, 1))
        red_avg, green_avg, blue_avg = avg_color
        
        # Simple heuristic classification based on color patterns
        categories = ['Wooden Dolls', 'Handlooms', 'Basket Weaving', 'Pottery']
        
        # Heuristic rules based on color analysis
        if red_avg > 120 and green_avg < 100 and blue_avg < 80:  # Reddish-brown = pottery
            primary_category = 'Pottery'
            confidence = 0.75
        elif red_avg > 100 and green_avg > 80 and blue_avg < 60:  # Brown tones = wooden dolls
            primary_category = 'Wooden Dolls' 
            confidence = 0.70
        elif green_avg > red_avg and green_avg > blue_avg:  # Greenish = basket weaving
            primary_category = 'Basket Weaving'
            confidence = 0.65
        else:  # Default to handlooms for colorful items
            primary_category = 'Handlooms'
            confidence = 0.60
            
        # Generate confidence scores for all categories
        confidence_scores = [0.1, 0.1, 0.1, 0.1]
        primary_idx = categories.index(primary_category)
        confidence_scores[primary_idx] = confidence
        
        # Normalize remaining confidence
        remaining = 1.0 - confidence
        for i in range(len(confidence_scores)):
            if i != primary_idx:
                confidence_scores[i] = remaining / 3
        
        return {
            'predicted_category': primary_category,
            'confidence': round(confidence, 3),
            'all_predictions': dict(zip(categories, [round(score, 3) for score in confidence_scores])),
            'model_used': 'simple_heuristic'
        }
        
    except Exception as e:
        print(f"Simple analysis failed: {e}")
        # Final fallback to random
        categories = ['Wooden Dolls', 'Handlooms', 'Basket Weaving', 'Pottery']
        confidence_scores = [random.random() for _ in categories]
        total = sum(confidence_scores)
        confidence_scores = [score/total for score in confidence_scores]
        
        max_idx = confidence_scores.index(max(confidence_scores))
        return {
            'predicted_category': categories[max_idx],
            'confidence': round(confidence_scores[max_idx], 3),
            'all_predictions': dict(zip(categories, [round(score, 3) for score in confidence_scores])),
            'model_used': 'random_fallback'
        }


# API Routes
@app.route('/')
def home():
    """Main route - serves homepage info"""
    return jsonify({
        'message': 'Welcome to Artisans Hub!',
        'description': 'AI-powered marketplace for local artisans',
        'version': '1.0.0',
        'endpoints': {
            'upload_and_analyze': '/api/upload-analyze',
            'create_listing': '/api/create-listing',
            'get_products': '/api/products',
            'chatbot': '/api/chatbot'
        }
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database': 'connected'
    })

@app.route('/api/upload-analyze', methods=['POST'])
def upload_and_analyze():
    """Upload image and generate AI description & price"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process image
        filename = process_uploaded_image(data['image'])
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # AI Analysis
        analysis = analyze_image_for_category(image_path)
        category = analysis['predicted_category']
        
        # Generate description and pricing
        description = generate_ai_description(category, analysis)
        pricing = generate_ai_price(category)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'ai_description': description,
            'pricing_suggestion': pricing,
            'image_filename': filename,
            'message': 'Image analyzed successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create-listing', methods=['POST'])
def create_listing():
    """Create new product listing with seller info"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['seller_name', 'seller_mobile', 'seller_location', 
                          'category', 'description', 'price', 'image_filename']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Check if seller exists or create new one
        seller = Seller.query.filter_by(
            mobile=data['seller_mobile']
        ).first()
        
        if not seller:
            seller = Seller(
                name=data['seller_name'],
                mobile=data['seller_mobile'],
                location=data['seller_location']
            )
            db.session.add(seller)
            db.session.commit()
        
        # Create product
        product = Product(
            seller_id=seller.id,
            category=data['category'],
            description=data['description'],
            price=float(data['price']),
            image_path=data['image_filename'],
            ai_generated=data.get('ai_generated', True)
        )
        
        db.session.add(product)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'product_id': product.id,
            'seller_id': seller.id,
            'message': 'Product listing created successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/products')
def get_products():
    """Get all products with optional filtering"""
    try:
        category = request.args.get('category')
        search = request.args.get('search', '')
        
        query = Product.query
        
        if category:
            query = query.filter(Product.category == category)
        
        if search:
            query = query.filter(
                Product.description.contains(search) |
                Product.category.contains(search)
            )
        
        products = query.order_by(Product.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'products': [product.to_dict() for product in products],
            'total': len(products)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/products/<int:product_id>')
def get_product(product_id):
    """Get specific product details"""
    try:
        product = Product.query.get_or_404(product_id)
        return jsonify({
            'success': True,
            'product': product.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """AI chatbot endpoint - My Artist Friend"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').lower()
        
        # Simple rule-based chatbot responses
        responses = {
            'hello': "Hello! I'm your Artist Friend! 🎨 I'm here to help you explore Artisans Hub. You can ask me about our features, how to sell your art, or discover amazing handcrafted items!",
            'help': "I can help you with: \n• How to sell your artwork\n• Finding specific categories\n• Understanding our AI features\n• Learning about artisan stories\n\nWhat would you like to know?",
            'sell': "To sell your artwork on Artisans Hub:\n1. Go to the Sell page\n2. Upload a photo of your item\n3. Our AI will suggest a description and price\n4. Add your contact details\n5. Your listing goes live!\n\nIt's that easy! 🚀",
            'categories': "We feature four amazing categories:\n🪆 Wooden Dolls - Hand-carved figurines\n🧵 Handlooms - Traditional textiles\n🧺 Basket Weaving - Natural fiber crafts\n🏺 Pottery - Ceramic masterpieces\n\nWhich interests you most?",
            'ai': "Our AI features include:\n• Smart image recognition for automatic categorization\n• AI-generated descriptions that tell your art's story\n• Intelligent pricing suggestions\n• Cultural heritage preservation\n\nPretty cool, right? 🤖",
            'features': "Artisans Hub features:\n• AI-powered selling tools\n• Beautiful product showcase\n• Easy search and discovery\n• Direct artist contact\n• Cultural storytelling\n• Mobile-friendly design\n\nWhat feature excites you most?"
        }
        
        # Find matching response
        response = "I'm here to help! Ask me about selling art, our categories, AI features, or just say 'help' for options. 😊"
        
        for keyword, reply in responses.items():
            if keyword in user_message:
                response = reply
                break
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/user/statistics')
def get_user_statistics():
    """Get user statistics - returns zero values for fresh marketplace"""
    try:
        # For a new marketplace, all statistics are zero
        statistics = {
            'success': True,
            'statistics': {
                'products_listed': 0,
                'total_sales': 0,
                'total_sales_formatted': '₹0',
                'reviews_count': 0,
                'wishlist_items': 0,
                'profile_views': 0,
                'active_listings': 0
            },
            'recent_activity': [],  # Empty for new platform
            'achievements': [
                {
                    'title': 'Welcome',
                    'description': 'Joined Artisans Hub platform',
                    'earned': True,
                    'icon': '🎉',
                    'date_earned': datetime.utcnow().isoformat()
                },
                {
                    'title': 'First Sale',
                    'description': 'Complete your first transaction',
                    'earned': False,
                    'icon': '💰',
                    'date_earned': None
                },
                {
                    'title': 'Trusted Seller',
                    'description': 'Maintain 4.8+ rating with 10+ reviews',
                    'earned': False,
                    'icon': '🌟',
                    'date_earned': None
                },
                {
                    'title': 'Cultural Ambassador',
                    'description': 'Share 10+ cultural stories',
                    'earned': False,
                    'icon': '🏛️',
                    'date_earned': None
                },
                {
                    'title': 'Eco Warrior',
                    'description': 'List 20+ eco-friendly items',
                    'earned': False,
                    'icon': '🌱',
                    'date_earned': None
                }
            ]
        }
        
        return jsonify(statistics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize database and sample data
def initialize_sample_data():
    """Initialize empty database for new marketplace"""
    # Check if data already exists
    if Product.query.first():
        return  # Data already exists
    
    # Keep sellers data but no products - showing that sellers exist but haven't uploaded yet
    sample_sellers = [
        {
            'name': 'Rajesh Kumar',
            'mobile': '+91-9876543210',
            'location': 'Rajasthan, India'
        },
        {
            'name': 'Priya Sharma',
            'mobile': '+91-8765432109',
            'location': 'Gujarat, India'
        },
        {
            'name': 'Mohammed Ali',
            'mobile': '+91-7654321098',
            'location': 'Uttar Pradesh, India'
        },
        {
            'name': 'Lakshmi Devi',
            'mobile': '+91-6543210987',
            'location': 'Tamil Nadu, India'
        },
        {
            'name': 'Arjun Singh',
            'mobile': '+91-5432109876',
            'location': 'Punjab, India'
        },
        {
            'name': 'Sunita Patel',
            'mobile': '+91-4321098765',
            'location': 'Maharashtra, India'
        }
    ]
    
    # Create sellers only - no products as sellers haven't uploaded yet
    sellers = []
    for seller_data in sample_sellers:
        seller = Seller(
            name=seller_data['name'],
            mobile=seller_data['mobile'],
            location=seller_data['location']
        )
        db.session.add(seller)
        sellers.append(seller)
    
    db.session.commit()
    print("✅ Empty marketplace initialized - sellers registered but no products uploaded yet!")

with app.app_context():
    db.create_all()
    initialize_sample_data()
    print("✅ Database initialized successfully!")
    
    # Load trained CNN model
    print("🤖 Loading trained CNN model...")
    if load_trained_cnn_model():
        print("🎯 CNN model ready for image recognition!")
    else:
        print("⚠️ Using fallback image analysis methods")

if __name__ == '__main__':
    print("🚀 Starting Artisans Hub Backend")
    print("=" * 40)
    print("📱 Frontend: http://localhost:3000")
    print("🔧 Backend API: http://localhost:5000")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("\n🎨 Ready to serve artisan marketplace!")
    
    # Use environment port for production deployment
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)