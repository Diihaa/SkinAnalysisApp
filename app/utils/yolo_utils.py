import os
import cv2
from ultralytics import YOLO

# Correct model path (make sure yolov8n.pt is in app/models)
MODEL_PATH = os.path.join("app", "models", "yolov8n.pt")
yolo_model = YOLO(MODEL_PATH)

def detect_acne(img_path):
    """
    Run YOLOv8 detection on an image.
    Currently uses general YOLOv8n model (not acne-specific).
    Replace yolov8n.pt with your custom acne-trained model later.
    """
    results = yolo_model(img_path, conf=0.4)  # Run YOLO inference
    detections = results[0].boxes
    acne_count = len(detections)

    # Map acne count to severity (demo thresholds)
    if acne_count == 0:
        severity = "None"
    elif acne_count <= 5:
        severity = "Mild"
    elif acne_count <= 15:
        severity = "Moderate"
    else:
        severity = "Severe"

    return acne_count, severity
