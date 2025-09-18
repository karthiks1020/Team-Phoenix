
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv

# Initialize extensions
db = SQLAlchemy()

def create_app():
    """Application factory function."""
    # Load environment variables from .env file
    load_dotenv(dotenv_path='../.env')

    app = Flask(__name__)

    # --- Configuration ---
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, '../uploads')

    # --- CORS Configuration ---
    if os.environ.get('FLASK_DEBUG') == '1':
        allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    else:
        allowed_origins = os.environ.get('CORS_ORIGINS', '*').split(',')
    CORS(app, origins=allowed_origins)

    # --- Initialize Extensions ---
    db.init_app(app)

    # --- Create Folders ---
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    with app.app_context():
        # --- Import and Register Blueprints ---
        from . import routes
        app.register_blueprint(routes.main_bp)
        print("✅ Blueprint registered.")

        # --- Create Database Tables ---
        print("🔨 Creating database tables...")
        db.create_all()
        print("✅ Database tables created.")

        # --- Load AI Model ---
        from . import ai_services
        print("🤖 Loading trained CNN model...")
        ai_services.load_trained_cnn_model()

    print("🎉 Flask app created successfully!")
    return app
