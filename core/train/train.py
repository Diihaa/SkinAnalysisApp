import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent / "dataset"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "validation"

# Save trained model directly into app/models
MODEL_DIR = Path(__file__).resolve().parents[2] / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === Image settings ===
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# === Data generators ===
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

# === Build CNN model ===
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax")
])

# === Compile ===
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# === Train ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# === Save model ===
model_path = MODEL_DIR / "cnn_skin_model.h5"
model.save(model_path)

print(f"✅ Model saved to {model_path}")
