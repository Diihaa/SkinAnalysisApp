# app/utils/preprocessing.py
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


# -----------------------------------------------------
# Face Cropping Utility
# -----------------------------------------------------
def crop_face(image: np.ndarray):
    """
    Detect and crop the largest face in the given image.
    If no face is found, return the original image (so pipeline continues).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load OpenCV Haarcascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # ⚠️ No face found → return original frame instead of None
        return image

    # Pick the largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_crop = image[y:y + h, x:x + w]
    return face_crop


# -----------------------------------------------------
# CNN Preprocessing
# -----------------------------------------------------
def prepare_image_for_cnn(image: np.ndarray, target_size=(128, 128)) -> np.ndarray:
    """
    Resize and normalize an image (face or full frame) for CNN.
    Returns shape (1, H, W, 3) with values in [0, 1].
    """
    img_resized = cv2.resize(image, target_size)
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -----------------------------------------------------
# ResNet Preprocessing
# -----------------------------------------------------
def prepare_image_for_resnet(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Resize and preprocess an image (face or full frame) for ResNet50.
    Returns array ready for ResNet50.predict().
    """
    img_resized = cv2.resize(image, target_size)
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
