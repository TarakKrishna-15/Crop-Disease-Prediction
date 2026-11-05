import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# ✅ Dataset path
DATASET_PATH = r"D:\CSP\data\plant_dataset"

# ✅ Save model path
SAVE_DIR = r"D:\CSP\saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "plant_disease_model.h5")

# ✅ Use only 1 subfolder (you can pick one: color, grayscale, segmented)
#   Example: use only 'color' if training is too slow
DATASET_PATH = os.path.join(DATASET_PATH, "color")

# ✅ Data preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),  # smaller image size = faster training
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# ✅ Lightweight CNN model (fast to train)
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train for fewer epochs
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5  # change to 3–5 for very fast training
)

# ✅ Save the model
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
