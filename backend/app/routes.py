
import os
import uuid
import base64
import io
from flask import Blueprint, request, jsonify, send_from_directory, current_app
from . import db
from .models import Seller, Product
from .ai_services import (
    cnn_image_analysis,
    generate_ai_description,
    generate_ai_price
)
from PIL import Image  # FIX: Import the Image module

# Define the Blueprint
main_bp = Blueprint('main_bp', __name__)

# --- Helper Function ---
def process_uploaded_image(image_data):
    """Decodes a base64 image, saves it, and returns the filename."""
    try:
        if 'data:image' in image_data:
            header, encoded = image_data.split(',', 1)
        else:
            encoded = image_data

        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        # Save the image
        image.save(filepath, 'JPEG', quality=85)
        return filename, filepath
    except Exception as e:
        current_app.logger.error(f"Failed to process image: {e}")
        raise

# --- API Routes ---
@main_bp.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serves an uploaded file."""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@main_bp.route('/api/upload-analyze', methods=['POST'])
def upload_and_analyze_route():
    """Receives an image, runs AI analysis, and returns results."""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        filename, image_path = process_uploaded_image(data['image'])

        analysis_results = cnn_image_analysis(image_path)
        category = analysis_results['predicted_category']

        description = generate_ai_description(category)
        pricing = generate_ai_price(category)

        return jsonify({
            'success': True,
            'analysis': analysis_results,
            'ai_description': description,
            'pricing_suggestion': pricing,
            'image_filename': filename
        })

    except Exception as e:
        current_app.logger.error(f"Upload and analyze failed: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500

@main_bp.route('/api/create-listing', methods=['POST'])
def create_listing_route():
    """Creates a new product listing."""
    data = request.get_json()
    required_fields = ['seller_name', 'seller_mobile', 'seller_location', 'category', 'description', 'price', 'image_filename']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    seller = Seller.query.filter_by(mobile=data['seller_mobile']).first()
    if not seller:
        seller = Seller(
            name=data['seller_name'],
            mobile=data['seller_mobile'],
            location=data['seller_location']
        )
        db.session.add(seller)
        db.session.commit()

    new_product = Product(
        seller_id=seller.id,
        category=data['category'],
        description=data['description'],
        price=float(data['price']),
        image_path=data['image_filename']
    )
    db.session.add(new_product)
    db.session.commit()

    return jsonify({
        'success': True,
        'product_id': new_product.id,
        'message': 'Product listing created successfully!'
    }), 201

@main_bp.route('/api/products', methods=['GET'])
def get_products_route():
    """Returns all product listings."""
    products = Product.query.order_by(Product.created_at.desc()).all()
    return jsonify([product.to_dict() for product in products])
