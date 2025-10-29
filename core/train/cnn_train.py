import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

# === Paths & Config ===
dataset_path = "data"  # Your dataset folder (should have subfolders: Acne, Dry, Normal, Oily)
img_size = 128
batch_size = 16
epochs = 20

# Save directly into app/models
MODEL_DIR = Path(__file__).resolve().parents[2] / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_save_path = MODEL_DIR / "cnn_skin_model.h5"

# === Data Augmentation ===
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# === Build CNN ===
model = Sequential([
    Input(shape=(img_size, img_size, 3)),

    Conv2D(32, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(train_data.num_classes, activation="softmax")  # auto adapts to # of classes (should be 4)
])

# === Compile ===
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# === Callbacks ===
checkpoint = ModelCheckpoint(
    model_save_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)

# === Train ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print("✅ CNN training completed. Model saved at:", model_save_path)
