import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

# Dataset path (update if different)
dataset_path = "../data/Rice_Image_Dataset"

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Build model
model = build_model(input_shape=(128,128,3), num_classes=train_generator.num_classes)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save model
model.save("../rice_cnn.h5")
print("✅ Model saved as rice_cnn.h5")
