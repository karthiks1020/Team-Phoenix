
import os
import uuid
import base64
import io
import re
import requests  # Added for making API calls
from flask import Blueprint, request, jsonify, send_from_directory, current_app
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
from . import db
from .models import Seller, Product
from .ai_services import (
    cnn_image_analysis,
    generate_ai_description,
    generate_ai_price
)
from PIL import Image

# Define the Blueprint
main_bp = Blueprint('main_bp', __name__)

# --- Helper Function ---
def process_uploaded_image(image_data):
    """Decodes a base64 image, saves it, uploads to S3 if configured, and returns (filename, local_path, image_url or None)."""
    try:
        if 'data:image' in image_data:
            header, encoded = image_data.split(',', 1)
        else:
            encoded = image_data

        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        image.save(filepath, 'JPEG', quality=85)

        # Optional S3 upload
        image_url = None
        bucket = os.environ.get('S3_BUCKET')
        region = os.environ.get('S3_REGION')
        custom_domain = os.environ.get('S3_CUSTOM_DOMAIN')
        if bucket and region:
            try:
                s3 = boto3.client('s3', region_name=region)
                s3.upload_file(
                    filepath,
                    bucket,
                    filename,
                    ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/jpeg'}
                )
                if custom_domain:
                    image_url = f"https://{custom_domain}/{filename}"
                else:
                    if region.startswith('cn-'):
                        base = f"https://{bucket}.s3.{region}.amazonaws.com.cn"
                    else:
                        base = f"https://{bucket}.s3.{region}.amazonaws.com"
                    image_url = f"{base}/{filename}"
            except (BotoCoreError, NoCredentialsError) as e:
                current_app.logger.error(f"S3 upload failed: {e}")
        return filename, filepath, image_url
    except Exception as e:
        current_app.logger.error(f"Failed to process image: {e}")
        raise

# --- NEW CHATBOT ROUTE ---
@main_bp.route('/api/chatbot', methods=['POST'])
def chatbot_route():
    """Handles chatbot messages by calling the Gemini API."""
    data = request.get_json()
    message = data.get('message')
    history = data.get('history', [])

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        current_app.logger.error('GEMINI_API_KEY is not set in environment.')
        return jsonify({'error': 'API key is not configured.'}), 500

    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_api_key}"

    system_prompt = """You are a friendly and helpful AI-powered chatbot assistant for a local artisan marketplace called 'Artisan Hub.' Your primary purpose is to assist local artisans and customers with questions about the platform. You must respond in a friendly and encouraging tone.

    You have the following knowledge base:

    - **Selling Artwork:** The platform helps local artisans market their craft, tell their stories, and expand their reach. To get started, artisans can create a profile and list their handmade items. The AI can assist with tasks like generating product descriptions.
    - **Product Categories:** The platform features several art categories including Wooden Dolls, Handlooms, Basket Weaving, and Pottery.
    - **AI Features:** The AI can help with tasks like generating product descriptions, suggesting ideal pricing, and translating product descriptions.
    - **Artisan Success Stories:** The marketplace plans to showcase success stories to inspire other sellers, but this feature is currently in development. You can mention that this is a great feature to add in the future.
    - **Platform Navigation:** The website has a user profile, a login page, a help center, and a search bar on the homepage."""

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": history + [{"role": "user", "parts": [{"text": message}]}],
    }

    try:
        response = requests.post(gemini_api_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        result = response.json()

        if 'candidates' not in result or not result['candidates']:
            current_app.logger.error("No response candidates found from Gemini API.")
            return jsonify({'error': 'Failed to get a valid response from the chatbot.'}), 500

        bot_response = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({'reply': bot_response})

    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"API call failed: {e}")
        return jsonify({'error': 'Failed to get a response from the chatbot.'}), 500
    except (KeyError, IndexError) as e:
        current_app.logger.error(f"Invalid response structure from Gemini API: {e}")
        return jsonify({'error': 'Failed to parse the chatbot response.'}), 500


# --- Existing API Routes ---
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

        filename, image_path, image_url = process_uploaded_image(data['image'])

        analysis_results = cnn_image_analysis(image_path)
        category = analysis_results['predicted_category']

        description = generate_ai_description(category)
        pricing = generate_ai_price(category)

        payload = {
            'success': True,
            'analysis': analysis_results,
            'ai_description': description,
            'pricing_suggestion': pricing,
            'image_filename': filename
        }
        if image_url:
            payload['image_url'] = image_url
        return jsonify(payload)

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

    # Validate mobile: exactly 10 digits
    mobile_digits = re.sub(r'\D', '', data['seller_mobile'])
    if not re.fullmatch(r'\d{10}', mobile_digits):
        return jsonify({'error': 'Invalid mobile number. Provide a 10-digit number.'}), 400

    seller = Seller.query.filter_by(mobile=mobile_digits).first()
    if not seller:
        seller = Seller(
            name=data['seller_name'],
            mobile=mobile_digits,
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
    """Returns product listings, optionally filtered by category, in a consistent structure."""
    try:
        category = request.args.get('category')
        query = Product.query
        if category:
            query = query.filter(Product.category == category)
        products = query.order_by(Product.created_at.desc()).all()
        return jsonify({
            'success': True,
            'products': [product.to_dict() for product in products]
        })
    except Exception as e:
        current_app.logger.error(f"Failed to retrieve products: {e}")
        return jsonify({'success': False, 'message': 'Internal Server Error'}), 500
