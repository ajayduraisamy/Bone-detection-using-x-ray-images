# train.py
# -----------------------------------
# Step 1: Import Libraries
# -----------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import numpy as np
import os

# -----------------------------------
# Step 2: Load & Preprocess CSV Data
# -----------------------------------
csv_file = "boneage.csv"
df = pd.read_csv(csv_file)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Convert to string for categorical classification
df["gender"] = df["gender"].astype(str)
df["age_category"] = df["age_category"].astype(str)

# -----------------------------------
# Step 3: Train/Validation Split
# -----------------------------------
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["age_category"]
)

# -----------------------------------
# Step 4: Image Generators
# -----------------------------------
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="age_category",
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=64,
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="age_category",
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=64,
    shuffle=False
)

num_classes = len(train_gen.class_indices)

# -----------------------------------
# Step 5: Model A -  CNN
# -----------------------------------
cnn_model = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("\n Training Custom CNN ...")
history_cnn = cnn_model.fit(train_gen, validation_data=val_gen, epochs=50)
cnn_val_acc = max(history_cnn.history["val_accuracy"])
cnn_model.save("cnn_model.h5")

# -----------------------------------
# Step 6: Model B - MobileNetV2 
# -----------------------------------
mobilenet_base = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights=None   # no pretrained weights
)

mobilenet_model = models.Sequential([
    mobilenet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

mobilenet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("\n Training MobileNetV2 ...")
history_mobilenet = mobilenet_model.fit(train_gen, validation_data=val_gen, epochs=10)
mobilenet_val_acc = max(history_mobilenet.history["val_accuracy"])
mobilenet_model.save("mobilenet_model.h5")

# -----------------------------------
# Step 7: Compare Models & Save Best
# -----------------------------------
if mobilenet_val_acc > cnn_val_acc:
    best_model = mobilenet_model
    best_name = "mobilenet_model.h5"
    best_acc = mobilenet_val_acc
else:
    best_model = cnn_model
    best_name = "cnn_model.h5"
    best_acc = cnn_val_acc

best_model.save("best_model.h5")
print(f"\n  Best model saved as best_model.h5 (from {best_name} with val_acc={best_acc:.4f})")

# -----------------------------------
# Step 8: Plot Training History (CNN vs MobileNetV2)
# -----------------------------------
plt.figure(figsize=(12,5))

# CNN Accuracy
plt.subplot(1,2,1)
plt.plot(history_cnn.history['val_accuracy'], label='CNN Val Accuracy')
plt.plot(history_mobilenet.history['val_accuracy'], label='MobileNetV2 Val Accuracy')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# CNN Loss
plt.subplot(1,2,2)
plt.plot(history_cnn.history['val_loss'], label='CNN Val Loss')
plt.plot(history_mobilenet.history['val_loss'], label='MobileNetV2 Val Loss')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

