import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. Dataset Parameters
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42
DATA_DIR = "./dataset"

print("Loading training and validation datasets...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"), # Change to "test" if your folder is named test
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# 2. Build CNN Model
print("Building CNN model...")
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3. Train Model
EPOCHS = 5
print("Starting training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 4. Save directly as .h5 file
model.save("chest_xray_model.h5")
print("Training complete! Model saved successfully as 'chest_xray_model.h5'.")