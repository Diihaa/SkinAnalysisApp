# test_cnn.py
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# === Paths ===
MODEL_PATH = os.path.join("app", "models", "cnn_skin_model.h5")
LABELS = ["Acne", "Dry", "Normal", "Oily"]  # must match your training classes

# === Load model ===
print("✅ Loading model...")
cnn_model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")


def preprocess_image(img_path, target_size=(128, 128)):
    """Load and preprocess image for CNN model"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(img_path):
    """Predict skin condition for given image"""
    img_array = preprocess_image(img_path)
    preds = cnn_model.predict(img_array, verbose=0)[0]

    # Best prediction
    idx = int(np.argmax(preds))
    best_label = LABELS[idx]
    confidence = round(float(preds[idx]) * 100, 2)

    # Print results
    print(f"\n📌 Prediction for {os.path.basename(img_path)}:")
    print(f"✅ Final Prediction: {best_label} ({confidence}%)\n")
    print("🔎 Full Probabilities:")
    for label, prob in zip(LABELS, preds):
        print(f"  {label}: {round(float(prob) * 100, 2)}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, help="Path to input image")
    args = parser.parse_args()

    predict(args.img_path)
