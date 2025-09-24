# test.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load Best Model
# ----------------------------
model = load_model("best_model.h5")
print("Model loaded successfully!")

# ----------------------------
# 2. Define Age Mapping
# ----------------------------
# Map your categorical age classes to representative ages (midpoints)

age_mapping = {
    "0": 1,
    "1": 4,
    "2": 7,
    "3": 10,
    "4": 13,
    "5": 16,
    "6": 19,
    "7": 22,
    "8": 25
}

class_labels = list(age_mapping.keys())

# ----------------------------
# 3. Preprocess Image
# ----------------------------
def preprocess_image(img_path, target_size=(224,224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# ----------------------------
# 4. Predict Age
# ----------------------------
def predict_age(img_path, model):
    img_array, img = preprocess_image(img_path)
    preds = model.predict(img_array)
    
    # Get predicted category
    class_idx = np.argmax(preds)
    predicted_cat = class_labels[class_idx]
    
    # Map to age
    predicted_age = age_mapping[predicted_cat]
    confidence = preds[0][class_idx]
    
    # Show image with prediction
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted Age: {predicted_age} yrs ({confidence*100:.2f}%)")
    plt.show()
    
    return predicted_age, confidence

# ----------------------------
# 5. File Dialog
# ----------------------------
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )

    if file_path:
        age, conf = predict_age(file_path, model)
        print(f"Predicted Age: {age} years (Confidence: {conf*100:.2f}%)")
    else:
        print("No file selected.")
