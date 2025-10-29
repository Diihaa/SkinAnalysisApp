# app/utils/cnn_utils.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load CNN model once (make sure the retrained 4-class model is saved here)
MODEL_PATH = os.path.join("app", "models", "cnn_skin_model.h5")
cnn_model = load_model(MODEL_PATH)

# Labels in the same order as your training classes (4 classes now)
labels = ["Acne", "Dry", "Normal", "Oily"]

def preprocess_image(img_path, target_size=(128, 128)):
    """Load and preprocess image for CNN model."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_skin_condition(img_path):
    """
    Run CNN model prediction on an image file path.
    Returns the best label and probability dict.
    """
    img_array = preprocess_image(img_path)
    preds = cnn_model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))
    best_label = labels[idx]
    # Probabilities in percentage
    probs = {label: round(float(p) * 100, 2) for label, p in zip(labels, preds)}
    return best_label, probs
