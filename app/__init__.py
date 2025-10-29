from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate   # ✅ added
from dotenv import load_dotenv
import os

# ML imports
import torch
from ultralytics import YOLO

db = SQLAlchemy()
login_manager = LoginManager()
migrate = None   # ✅ added placeholder

# Globals to share across app
YOLO_MODEL = None
DEVICE = "cpu"

def create_app():
    global YOLO_MODEL, DEVICE, migrate   # ✅ include migrate in globals

    # Load .env file
    load_dotenv()

    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Load config from .env
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

    # Init extensions
    db.init_app(app)
    migrate = Migrate(app, db)   # ✅ initialize Flask-Migrate
    login_manager.init_app(app)
    login_manager.login_view = "main.login"
    login_manager.login_message_category = "info"

    # -------------------------
    # YOLO Model Initialization
    # -------------------------
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)  # reduce CPU contention
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    if YOLO_MODEL is None:
        # Use your custom model or fallback to nano for speed
        weights_path = os.getenv("YOLO_WEIGHTS", "weights/best.pt")
        YOLO_MODEL = YOLO(weights_path)
        YOLO_MODEL.to(DEVICE)
        print(f"✅ YOLO model loaded on {DEVICE}: {weights_path}")

    # Register blueprint
    from app import routes
    app.register_blueprint(routes.bp)

    # Import models for Flask-Login
    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    return app