
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import text

# Initialize extensions
db = SQLAlchemy()

def create_app():
    """Application factory function."""
    # Load environment variables from .env file
    load_dotenv(dotenv_path='../.env')

    app = Flask(__name__)

    # --- Configuration ---
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')

    # Normalize DATABASE_URL (Render/Railway sometimes provide postgres:// which SQLAlchemy expects as postgresql://)
    db_url = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, '../uploads')

    # Set debug mode based on environment variable
    app.debug = os.environ.get('FLASK_DEBUG') == '1'

    # --- CORS Configuration ---
    allowed_origins = []
    if app.debug:
        # Explicitly allow frontend development origins when in debug mode
        allowed_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])
        # Also include the database host if it's relevant for CORS, though usually not for frontend
        allowed_origins.append("http://localhost:3306") # This might be for other internal calls, keep it for now
    
    # If CORS_ORIGINS is set, use it, otherwise default to '*' if not in debug or if debug doesn't cover it
    cors_env_origins = os.environ.get('CORS_ORIGINS')
    if cors_env_origins:
        allowed_origins.extend(cors_env_origins.split(','))
    elif not app.debug: # If not in debug and CORS_ORIGINS not set, allow all
        allowed_origins.append('*')

    # Remove duplicates and ensure unique origins
    allowed_origins = list(set(allowed_origins))
    
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

        # --- Test DB Connection & Create Tables ---
        try:
            print("🔌 Testing database connection...")
            with db.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
            print("✅ Database connection OK")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise

        print("🔨 Creating database tables...")
        db.create_all()
        print("✅ Database tables created.")

        # --- Load AI Model ---
        from . import ai_services
        print("🤖 Loading trained CNN model...")
        ai_services.load_trained_cnn_model()

    print("🎉 Flask app created successfully!")
    return app
