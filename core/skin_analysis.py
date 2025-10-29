# core/skin_analysis.py
from pathlib import Path
import sys
import cv2
import numpy as np
from datetime import datetime
import os, webbrowser
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# --- Ensure project root is in sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# TensorFlow / Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# YOLOv8
from ultralytics import YOLO

# ✅ imports from app.utils
from app.utils.preprocessing import crop_face, prepare_image_for_cnn
from app.utils.cnn_utils import predict_skin_condition

# ---------------------------------------------------------------------
# Model locations
# ---------------------------------------------------------------------
MODEL_DIR = ROOT / "app" / "models"
CNN_PATH  = MODEL_DIR / "cnn_skin_model.h5"
YOLO_PATH = MODEL_DIR / "yolov8n.pt"

# Labels used by CNN
CNN_LABELS = ["Acne", "Dry", "Normal", "Oily", "Sensitive"]

# Lazy singletons
_CNN_MODEL = None
_RESNET    = None
_YOLO      = None


def _get_cnn():
    global _CNN_MODEL
    if _CNN_MODEL is None:
        if not CNN_PATH.exists():
            raise FileNotFoundError(f"CNN model not found: {CNN_PATH}")
        _CNN_MODEL = load_model(str(CNN_PATH))
    return _CNN_MODEL


def _get_resnet():
    global _RESNET
    if _RESNET is None:
        _RESNET = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return _RESNET


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        if not YOLO_PATH.exists():
            raise FileNotFoundError(f"YOLO weights not found: {YOLO_PATH}")
        _YOLO = YOLO(str(YOLO_PATH))
    return _YOLO


def _predict_skin_type_resnet(face_bgr: np.ndarray) -> tuple:
    resnet = _get_resnet()
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (224, 224))
    arr = img_to_array(face_rgb)[None, ...]
    arr = preprocess_input(arr)
    feats = resnet.predict(arr, verbose=0)
    score = float(np.mean(feats))

    if score < 1.0:
        return "Dry", round((1.0 - score) * 100, 2)
    elif score > 1.5:
        return "Oily", round((score / 2.0) * 100, 2)
    else:
        return "Normal", round((1.0 - abs(score - 1.2)) * 100, 2)


def _predict_cnn_condition(face_bgr: np.ndarray):
    model = _get_cnn()
    x = prepare_image_for_cnn(face_bgr)
    preds = model.predict(x, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    best_label = CNN_LABELS[best_idx]
    best_conf = round(preds[best_idx] * 100, 2)

    probs = {label: round(float(p) * 100, 2) for label, p in zip(CNN_LABELS, preds)}
    return best_label, best_conf, probs


def _detect_acne_count(img_bgr: np.ndarray) -> int:
    yolo = _get_yolo()
    results = yolo(img_bgr, verbose=False)[0]
    return len(results.boxes)


def _acne_severity_from_count(count: int) -> str:
    if count < 2:
        return "Mild"
    if count < 5:
        return "Moderate"
    return "Severe"


# ---------------- PDF Report Generator ----------------
def generate_pdf_report(results: dict, cnn_probs: dict, img, final_pred_label, final_pred_conf):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Skin_Analysis_Report_{now}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 50, "DD AURA - Skin Analysis Report")

    # Save temporary face image for embedding
    img_filename = f"temp_face_{now}.jpg"
    cv2.imwrite(img_filename, img)

    # Add image with prediction label
    img_reader = ImageReader(img_filename)
    c.drawImage(img_reader, 180, height - 350, width=250, height=250)
    c.setFont("Helvetica", 12)
    c.drawString(180, height - 370, f"{final_pred_label} ({final_pred_conf}%)")

    # Final Prediction
    y = height - 420
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Final Prediction: {final_pred_label}")
    y -= 20

    # Probabilities
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Probabilities:")
    y -= 20
    for label, prob in cnn_probs.items():
        c.drawString(70, y, f"{label}: {prob}%")
        y -= 20

    # Acne and ResNet info
    y -= 20
    for key, value in results.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    c.save()
    os.remove(img_filename)

    try:
        os.startfile(filename)
    except:
        webbrowser.open_new(filename)


# ---------------- Main Analysis ----------------
def analyze_image(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Cannot read image: {image_path}"}

    face = crop_face(img)
    if face is None:
        face = img

    # ResNet
    skin_type_resnet, skin_conf = _predict_skin_type_resnet(face)

    # CNN
    cnn_condition, cnn_conf, cnn_probs = _predict_cnn_condition(face)

    # YOLO Acne
    acne_count = _detect_acne_count(img)
    acne_severity = _acne_severity_from_count(acne_count)

    results = {
        "Skin Type (ResNet)": f"{skin_type_resnet} ({skin_conf}%)",
        "Acne Count": int(acne_count),
        "Acne Severity": acne_severity
    }

    # ✅ Generate PDF with face image, label, probs, final prediction
    generate_pdf_report(results, cnn_probs, face, cnn_condition, cnn_conf)

    return results
