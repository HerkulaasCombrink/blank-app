import streamlit as st
import os
import zipfile
import cv2
import numpy as np
import tempfile
import json
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate, Blur

# Define augmentation pipeline for images
def augment_image(image):
    augmentations = Compose([
        RandomBrightnessContrast(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
        Blur(blur_limit=3, p=0.3)
    ])
    augmented = augmentations(image=image)
    return augmented['image']

# Process images
def generate_synthetic_images(uploaded_file, label):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    synthetic_images = [augment_image(image) for _ in range(10)]
    return synthetic_images, label

# Create ZIP archive
def create_zip(files, metadata, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))
        metadata_path = "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        zipf.write(metadata_path, "metadata.json")

# Streamlit UI
st.title("Synthetic Data Generator for Sign Language")
option = st.radio("Choose Data Type", ("Image", "Video"))
uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "mp4"])
label = st.text_input("Enter Label for the Data")

if uploaded_file and label:
    if option == "Image":
        st.image(uploaded_file, caption="Original Image", use_column_width=True)
        if st.button("Generate Synthetic Images"):
            synthetic_images, label = generate_synthetic_images(uploaded_file, label)
            image_files = []
            metadata = {"label": label, "images": []}
            
            for i, img in enumerate(synthetic_images):
                img_path = f"synthetic_image_{i}.jpg"
                cv2.imwrite(img_path, img)
                image_files.append(img_path)
                metadata["images"].append({"file": img_path, "label": label})
                st.image(img, caption=f"Synthetic Image {i+1}", use_column_width=True)
            
            zip_name = "synthetic_images.zip"
            create_zip(image_files, metadata, zip_name)
            st.download_button("Download ZIP", data=open(zip_name, 'rb').read(), file_name=zip_name)
