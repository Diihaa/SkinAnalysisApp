import os
import cv2
import numpy as np
import datetime
import time
import json
import pytz
from flask import Blueprint, render_template, redirect, url_for, flash, request, Response, jsonify, send_file
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, YOLO_MODEL, DEVICE
from app.models import User, AnalysisRecord

# Import utils
from app.utils.preprocessing import crop_face
from app.utils.cnn_utils import predict_skin_condition

bp = Blueprint("main", __name__)

# ------------------------------
# Globals for camera handling
# ------------------------------
cap = None
last_frame = None

# ------------------------------
# Home Page (with live counters)
# ------------------------------
@bp.route("/")
def home():
    user_count = User.query.count()
    scan_count = AnalysisRecord.query.count()
    return render_template("home.html", user_count=user_count, scan_count=scan_count)

# ------------------------------
# About Page
# ------------------------------
@bp.route("/about")
def about():
    return render_template("about.html")

# ------------------------------
# Help Page
# ------------------------------
@bp.route("/help")
def help():
    return render_template("help.html")

# ------------------------------
# Register Page
# ------------------------------
@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "danger")
            return redirect(url_for("main.register"))

        new_user = User(
            name=name,
            email=email,
            password=generate_password_hash(password, method="pbkdf2:sha256")  # ✅ FIXED
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("main.login"))

    return render_template("register.html")

# ------------------------------
# Login Page
# ------------------------------
@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("main.dashboard"))
        else:
            flash("Invalid email or password", "danger")
            return redirect(url_for("main.login"))

    return render_template("login.html")

# ------------------------------
# Logout
# ------------------------------
@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.login"))

# ------------------------------
# Dashboard
# ------------------------------
@bp.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)

# ------------------------------
# Video feed generator
# ------------------------------
def gen():
    global cap, last_frame
    if cap is None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()
        frame = cv2.flip(frame, 1)

        # Encode frame for streaming
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@bp.route("/video_feed")
@login_required
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------------------------
# Stop Scan → run analysis
# ------------------------------
@bp.route("/stop_scan")
@login_required
def stop_scan():
    global cap, last_frame

    if cap:
        cap.release()
        cap = None

    if last_frame is None:
        flash("No frame captured!", "danger")
        return redirect(url_for("main.dashboard"))

    # ---- 1. Crop face ----
    face_crop = crop_face(last_frame)
    if face_crop is None:
        flash("No face detected!", "danger")
        return redirect(url_for("main.dashboard"))

    filename = f"user_{current_user.id}_{int(time.time())}.jpg"
    filepath = os.path.join("app", "static", "captures", filename)
    cv2.imwrite(filepath, face_crop)

    # ---- 2. CNN Prediction ----
    skin_type, probs = predict_skin_condition(filepath)
    final_pred = skin_type

    # ---- 3. YOLO Acne detection ----
    results = YOLO_MODEL.predict(last_frame, imgsz=320, conf=0.3, device=DEVICE, verbose=False)
    acne_count = len(results[0].boxes)
    severity = "Mild" if acne_count <= 3 else "Moderate" if acne_count <= 10 else "Severe"

    # ---- 4. Save to DB ----
    record = AnalysisRecord(
        user_id=current_user.id,
        filename=filename,
        skin_type=final_pred,
        cnn_condition=final_pred,
        acne_count=acne_count,
        acne_severity=severity,
        probabilities=json.dumps(probs),
        created_at=datetime.datetime.utcnow()
    )
    db.session.add(record)
    db.session.commit()

    return redirect(url_for("main.results"))

# ------------------------------
# Results Page
# ------------------------------
@bp.route("/results")
@login_required
def results():
    latest_record = AnalysisRecord.query.filter_by(user_id=current_user.id).order_by(
        AnalysisRecord.created_at.desc()
    ).first()

    probs = None
    if latest_record and latest_record.probabilities:
        try:
            probs = json.loads(latest_record.probabilities)
        except Exception:
            probs = None

    # ✅ Product Recommendations
    product_recommendations = {
        "Oily": [
            {
                "name": "Minimalist 10% Niacinamide Serum",
                "img": "images/minimalist_serum.jpg",
                "link": "https://www.amazon.in/dp/B08XYN7X8L?tag=skinnoura-21"
            },
            {
                "name": "Neutrogena Oil-Free Acne Wash",
                "img": "images/neutrogena_cleanser.jpg",
                "link": "https://www.amazon.in/dp/B01M8L5Z3Y?tag=skinnoura-21"
            }
        ],
        "Dry": [
            {
                "name": "Cetaphil Moisturizing Cream",
                "img": "images/cetaphil.jpg",
                "link": "https://www.amazon.in/dp/B00EYTM9SU?tag=skinnoura-21"
            },
            {
                "name": "Mamaearth Oil-Free Moisturizer",
                "img": "images/mamaearth_moisturizer.jpg",
                "link": "https://www.amazon.in/dp/B07H9MCW3V?tag=skinnoura-21"
            }
        ],
        "Sensitive": [
            {
                "name": "Aveeno Calm + Restore Oat Gel",
                "img": "images/aveeno.jpg",
                "link": "https://www.amazon.in/dp/B08M8L5Z9T?tag=skinnoura-21"
            },
            {
                "name": "La Roche-Posay Thermal Spring Water",
                "img": "images/laroche.jpg",
                "link": "https://www.amazon.in/dp/B00D5L4N78?tag=skinnoura-21"
            }
        ],
        "Combination": [
            {
                "name": "The Ordinary Niacinamide 10% + Zinc",
                "img": "images/ordinary.jpg",
                "link": "https://www.amazon.in/dp/B08KX6DL7X?tag=skinnoura-21"
            },
            {
                "name": "Plum Green Tea Night Gel",
                "img": "images/plum.jpg",
                "link": "https://www.amazon.in/dp/B00PAZDH8O?tag=skinnoura-21"
            }
        ]
    }

    # ✅ Decide which products to recommend
    recommended_products = []
    if latest_record:
        skin_type = latest_record.cnn_condition or latest_record.skin_type
        recommended_products = product_recommendations.get(skin_type, [])

        # Convert UTC to IST
        ist = pytz.timezone("Asia/Kolkata")
        latest_record.local_time = latest_record.created_at.replace(tzinfo=pytz.utc).astimezone(ist)

    return render_template(
        "results.html",
        record=latest_record,
        probs=probs,
        products=recommended_products
    )

# ------------------------------
# My Records Page (History)
# ------------------------------
@bp.route("/my_records")
@login_required
def my_records():
    records = AnalysisRecord.query.filter_by(user_id=current_user.id).order_by(
        AnalysisRecord.created_at.desc()
    ).all()

    ist = pytz.timezone("Asia/Kolkata")
    for r in records:
        try:
            r.parsed_probs = json.loads(r.probabilities) if r.probabilities else {}
        except Exception:
            r.parsed_probs = {}
        r.local_time = r.created_at.replace(tzinfo=pytz.utc).astimezone(ist)

    return render_template("my_records.html", records=records)

# ------------------------------
# Download PDF Report
# ------------------------------
@bp.route("/download_report/<int:record_id>")
@login_required
def download_report(record_id):
    record = AnalysisRecord.query.filter_by(id=record_id, user_id=current_user.id).first()
    if not record:
        flash("Report not found!", "danger")
        return redirect(url_for("main.results"))

    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "DD AURA - Skin Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Show Image if available
    if record.filename:
        img_path = os.path.join("app", "static", "captures", record.filename)
        if os.path.exists(img_path):
            pdf.image(img_path, x=70, w=70)
            pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Final Skin Type: {record.skin_type}", ln=True)
    pdf.cell(0, 10, f"Acne Count: {record.acne_count}", ln=True)
    pdf.cell(0, 10, f"Acne Severity: {record.acne_severity}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Probabilities:", ln=True)

    probs = json.loads(record.probabilities)
    pdf.set_font("Arial", "", 12)
    for k, v in probs.items():
        pdf.cell(0, 10, f"{k}: {round(v, 2)}%", ln=True)

    # Add IST timestamp
    ist = pytz.timezone("Asia/Kolkata")
    local_time = record.created_at.replace(tzinfo=pytz.utc).astimezone(ist)
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Generated on: {local_time.strftime('%d %b %Y, %I:%M %p IST')}", ln=True)

    filename = f"SkinReport_{record.created_at.strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join("app", "static", "reports", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath)

    return send_file(filepath, as_attachment=True)

# ------------------------------
# Search Results Page (dynamic)
# ------------------------------
@bp.route("/search_results")
@login_required
def search_results():
    query = request.args.get("q", "").strip()
    results = []

    if query:
        results = AnalysisRecord.query.filter(
            (AnalysisRecord.user_id == current_user.id) &
            (
                AnalysisRecord.skin_type.ilike(f"%{query}%") |
                AnalysisRecord.acne_severity.ilike(f"%{query}%") |
                AnalysisRecord.cnn_condition.ilike(f"%{query}%")
            )
        ).order_by(AnalysisRecord.created_at.desc()).all()

    # Add IST conversion
    ist = pytz.timezone("Asia/Kolkata")
    for r in results:
        r.local_time = r.created_at.replace(tzinfo=pytz.utc).astimezone(ist)

    return render_template("search_results.html", query=query, results=results)
