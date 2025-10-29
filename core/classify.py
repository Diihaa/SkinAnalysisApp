import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("../server/dataset/models/skin_model.h5")

# Class labels (update to match your dataset)
class_labels = ["acne", "dry", "oily"]

def classify(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    return predicted_class, confidence
