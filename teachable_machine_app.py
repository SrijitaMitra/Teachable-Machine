import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import shutil

st.set_page_config(page_title="Teachable Machine", layout="centered")
st.title("🧠 Teachable Machine")
st.markdown("Train your own image classifier right in the browser.")

IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 5
MODEL_PATH = "teachable_model.h5"
EXTRACTION_DIR = "extracted_data"

# Clean and prepare the extraction directory
if os.path.exists(EXTRACTION_DIR):
    shutil.rmtree(EXTRACTION_DIR)
os.makedirs(EXTRACTION_DIR, exist_ok=True)

# Step 1: Upload Dataset
st.header("1. Upload Training Data")
st.markdown("Upload a ZIP file with class folders (e.g., `happy/`, `sad/`, etc.) containing images.")
train_zip = st.file_uploader("Upload Zip", type=["zip"])
train_dir = None
show_train_button = False

if train_zip is not None:
    zip_path = os.path.join(EXTRACTION_DIR, "train.zip")
    with open(zip_path, "wb") as f:
        f.write(train_zip.read())
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTION_DIR)
    subdirs = [os.path.join(EXTRACTION_DIR, d) for d in os.listdir(EXTRACTION_DIR) if os.path.isdir(os.path.join(EXTRACTION_DIR, d)) and d != '__MACOSX']
    if subdirs:
        train_dir = EXTRACTION_DIR
        show_train_button = True
        st.success("✅ Dataset uploaded and extracted successfully.")
    else:
        st.warning("⚠️ ZIP does not contain class folders with images. Please check your ZIP structure.")
else:
    st.info("📂 Please upload a ZIP file to begin training.")

# Step 2: Train the Model
model = None
if train_dir and show_train_button:
    st.header("2. Train the Model")
    if st.button("Start Training"):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        val_gen = datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )

        num_classes = len(train_gen.class_indices)
        class_labels = list(train_gen.class_indices.keys())

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        with st.spinner("Training in progress..."):
            history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

        # Save model
        model.save(MODEL_PATH)
        st.success("Model trained and saved successfully!")

        # Plot training history
        st.subheader("📊 Training Accuracy and Loss")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['accuracy'], label='Train Acc')
        ax[0].plot(history.history['val_accuracy'], label='Val Acc')
        ax[0].set_title('Accuracy')
        ax[0].legend()

        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Val Loss')
        ax[1].set_title('Loss')
        ax[1].legend()

        st.pyplot(fig)

        st.session_state['model'] = model
        st.session_state['class_labels'] = class_labels

# Step 3: Predict
if 'model' in st.session_state or os.path.exists(MODEL_PATH):
    st.header("3. Predict")
    test_img = st.file_uploader("Upload Image for Prediction", type=["jpg", "jpeg", "png"])
    if test_img is not None:
        image = Image.open(test_img).convert("RGB")
        image = image.resize(IMG_SIZE)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        if 'model' not in st.session_state:
            model = load_model(MODEL_PATH)
            st.session_state['model'] = model
            st.session_state['class_labels'] = [f"Class {i}" for i in range(model.output_shape[1])]

        prediction = st.session_state['model'].predict(image_array)
        if isinstance(st.session_state['class_labels'], list):
            predicted_class = st.session_state['class_labels'][np.argmax(prediction)]
        else:
            predicted_class = f"Class {np.argmax(prediction)}"
        confidence = np.max(prediction)

        st.image(test_img, caption="Uploaded Image", use_column_width=True)
        st.success(f"Prediction: {predicted_class} ({confidence*100:.2f}% confidence)")
