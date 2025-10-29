from app import db
from datetime import datetime
from flask_login import UserMixin

# === User model for login/register ===
class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # store hashed password
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship → one user can have many analysis records
    records = db.relationship("AnalysisRecord", backref="user", lazy=True)

    def __repr__(self):
        return f"<User {self.email}>"


# === Analysis Record model ===
class AnalysisRecord(db.Model):
    __tablename__ = "analysis_records"

    id = db.Column(db.Integer, primary_key=True)

    # Store captured image filename
    filename = db.Column(db.String(255), nullable=True)

    # Predicted labels
    skin_type = db.Column(db.String(100))
    cnn_condition = db.Column(db.String(100))  # e.g., acne, oily, dry
    acne_count = db.Column(db.Integer)
    acne_severity = db.Column(db.String(50))

    # Store probabilities as JSON
    probabilities = db.Column(db.JSON)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Foreign key → link record to a user
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    # Convenience property: combine condition info for search
    @property
    def condition(self):
        return f"{self.skin_type or ''} {self.cnn_condition or ''} {self.acne_severity or ''}".strip()

    def __repr__(self):
        return f"<AnalysisRecord {self.filename or 'no_file'} - {self.skin_type}>"

# === Product model (future-proof for recommendations) ===
class Product(db.Model):
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(100))  # e.g., "acne care", "oily skin"
    price = db.Column(db.Float)
    link = db.Column(db.String(500))  # affiliate link (Amazon/Flipkart)
    image_url = db.Column(db.String(500))  # product image
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Product {self.name} - {self.category}>"

# === Testimonial model (optional, for homepage reviews) ===
class Testimonial(db.Model):
    __tablename__ = "testimonials"

    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100))
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Testimonial {self.user_name}>"
