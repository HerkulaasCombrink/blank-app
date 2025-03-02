import streamlit as st
import os
import zipfile
import cv2
import numpy as np
import tempfile
import json
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate, Blur, ElasticTransform, GridDistortion, OpticalDistortion, HueSaturationValue

# Define augmentation pipeline for diverse transformations with color and face distortions
def augment_image(image):
    augmentations = Compose([
        RandomBrightnessContrast(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.9),
        Blur(blur_limit=5, p=0.5),
        ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.7),
        GridDistortion(num_steps=5, distort_limit=0.3, p=0.6),
        OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.6),
        HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=50, p=0.7)  # Color variation
    ])
    augmented = augmentations(image=image)
    return augmented['image']

# Overlay label on image
def overlay_label(image, label):
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = image.shape[0] - 20
    cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return overlay

# Annotate synthetic images
def annotate_image(image):
    st.image(image, caption="Synthetic Image for Annotation", use_column_width=True)
    annotation = st.text_input("Enter annotation for this image")
    return annotation

# Process images with adjustable number of synthetic samples
def generate_synthetic_images(uploaded_file, label, num_images):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    synthetic_images = [augment_image(image) for _ in range(num_images)]
    labeled_images = [overlay_label(img, label) for img in synthetic_images]
    return labeled_images, label

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
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg"])
label = st.text_input("Enter Label for the Data")
num_images = st.slider("Number of Synthetic Images", min_value=1, max_value=20, value=10)

if uploaded_file and label:
    st.image(uploaded_file, caption="Original Image", use_column_width=True)
    if st.button("Generate Synthetic Images"):
        synthetic_images, label = generate_synthetic_images(uploaded_file, label, num_images)
        image_files = []
        metadata = {"label": label, "images": []}
        
        for i, img in enumerate(synthetic_images):
            annotation = annotate_image(img)
            img_path = f"synthetic_image_{i}.jpg"
            cv2.imwrite(img_path, img)
            image_files.append(img_path)
            metadata["images"].append({"file": img_path, "label": label, "annotation": annotation})
            st.image(img, caption=f"Synthetic Image {i+1}", use_column_width=True)
        
        zip_name = "synthetic_images.zip"
        create_zip(image_files, metadata, zip_name)
        st.download_button("Download ZIP", data=open(zip_name, 'rb').read(), file_name=zip_name)
