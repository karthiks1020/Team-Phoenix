
from . import db
from datetime import datetime

class Seller(db.Model):
    """Represents a seller in the marketplace."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    mobile = db.Column(db.String(20), unique=True, nullable=False)
    location = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to Product model
    products = db.relationship('Product', backref='seller', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        """Serializes the object to a dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'mobile': self.mobile,
            'location': self.location,
            'created_at': self.created_at.isoformat()
        }

class Product(db.Model):
    """Represents a product listing in the marketplace."""
    id = db.Column(db.Integer, primary_key=True)
    seller_id = db.Column(db.Integer, db.ForeignKey('seller.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    ai_generated = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Serializes the object to a dictionary."""
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
