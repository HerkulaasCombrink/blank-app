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

# Draw bounding boxes on image for hand and face
def draw_bounding_boxes(image, face_coords, hand_coords):
    overlay = image.copy()
    if face_coords:
        x1, y1, x2, y2 = face_coords
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(overlay, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if hand_coords:
        x1, y1, x2, y2 = hand_coords
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(overlay, "Hand", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return overlay

# Process images with adjustable number of synthetic samples
def generate_synthetic_images(image, face_coords, hand_coords, num_images):
    synthetic_images = [augment_image(image) for _ in range(num_images)]
    labeled_images = [draw_bounding_boxes(img, face_coords, hand_coords) for img in synthetic_images]
    return labeled_images

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
num_images = st.slider("Number of Synthetic Images", min_value=1, max_value=20, value=10)

if uploaded_file:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Interactive bounding box selection
    st.write("**Adjust Face and Hand Bounding Boxes**")
    face_x1 = st.slider("Face X1", 0, image.shape[1], int(image.shape[1] * 0.3))
    face_y1 = st.slider("Face Y1", 0, image.shape[0], int(image.shape[0] * 0.2))
    face_x2 = st.slider("Face X2", 0, image.shape[1], int(image.shape[1] * 0.7))
    face_y2 = st.slider("Face Y2", 0, image.shape[0], int(image.shape[0] * 0.5))
    hand_x1 = st.slider("Hand X1", 0, image.shape[1], int(image.shape[1] * 0.3))
    hand_y1 = st.slider("Hand Y1", 0, image.shape[0], int(image.shape[0] * 0.6))
    hand_x2 = st.slider("Hand X2", 0, image.shape[1], int(image.shape[1] * 0.7))
    hand_y2 = st.slider("Hand Y2", 0, image.shape[0], int(image.shape[0] * 0.9))
    
    # Preview bounding boxes on the original image
    face_coords = (face_x1, face_y1, face_x2, face_y2)
    hand_coords = (hand_x1, hand_y1, hand_x2, hand_y2)
    preview_image = draw_bounding_boxes(image, face_coords, hand_coords)
    st.image(preview_image, caption="Preview Bounding Boxes", use_column_width=True)
    
    if st.button("Generate Synthetic Images"):
        synthetic_images = generate_synthetic_images(image, face_coords, hand_coords, num_images)
        
        image_files = []
        metadata = {"images": []}
        
        for i, img in enumerate(synthetic_images):
            img_path = f"synthetic_image_{i}.jpg"
            cv2.imwrite(img_path, img)
            image_files.append(img_path)
            metadata["images"].append({"file": img_path, "face_bbox": face_coords, "hand_bbox": hand_coords})
            st.image(img, caption=f"Synthetic Image {i+1}", use_column_width=True)
        
        zip_name = "synthetic_images.zip"
        create_zip(image_files, metadata, zip_name)
        st.download_button("Download ZIP", data=open(zip_name, 'rb').read(), file_name=zip_name)
