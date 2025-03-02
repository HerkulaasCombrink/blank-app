import streamlit as st
import os
import zipfile
import cv2
import numpy as np
import tempfile
import json
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate, Blur, ElasticTransform, GridDistortion, OpticalDistortion, HueSaturationValue, BboxParams

# Define augmentation pipeline with bounding box support
def augment_image_with_bboxes(image, bboxes):
    category_labels = ["face", "hand"]
    if len(bboxes) == 3:
        category_labels.append("hand2")
    
    augmentations = Compose([
        RandomBrightnessContrast(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        Blur(blur_limit=5, p=0.5),
        ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.7),
        GridDistortion(num_steps=5, distort_limit=0.3, p=0.6),
        OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.6),
        HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=50, p=0.7)  # Color variation
    ], bbox_params=BboxParams(format='pascal_voc', label_fields=['category']))
    augmented = augmentations(image=image, bboxes=bboxes, category=category_labels)
    return augmented['image'], augmented['bboxes']

# Draw bounding boxes on image
def draw_bounding_boxes(image, bboxes):
    overlay = image.copy()
    labels = ["Face", "Hand"]
    if len(bboxes) == 3:
        labels.append("Hand2")
    
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Red, Blue, Green
    for bbox, label, color in zip(bboxes, labels, colors):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return overlay

# Process images with bounding box tracking
def generate_synthetic_images(image, bboxes, num_images):
    synthetic_images = []
    updated_bboxes = []
    
    for _ in range(num_images):
        aug_img, aug_bboxes = augment_image_with_bboxes(image, bboxes)
        synthetic_images.append(aug_img)
        updated_bboxes.append(aug_bboxes)
    
    labeled_images = [draw_bounding_boxes(img, bboxes) for img, bboxes in zip(synthetic_images, updated_bboxes)]
    return labeled_images, updated_bboxes

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
    st.image(image, caption="Preview Bounding Boxes", use_column_width=True)
    
    option = st.radio("Select Type of Sign", ["One-Handed Sign", "Two-Handed Sign"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Adjust Hand 1 Bounding Box**")
        hand_x1 = st.slider("Hand 1 X1", 0, image.shape[1], int(image.shape[1] * 0.3))
        hand_y1 = st.slider("Hand 1 Y1", 0, image.shape[0], int(image.shape[0] * 0.6))
        hand_x2 = st.slider("Hand 1 X2", 0, image.shape[1], int(image.shape[1] * 0.7))
        hand_y2 = st.slider("Hand 1 Y2", 0, image.shape[0], int(image.shape[0] * 0.9))
    
    with col2:
        st.write("**Adjust Face Bounding Box**")
        face_x1 = st.slider("Face X1", 0, image.shape[1], int(image.shape[1] * 0.3))
        face_y1 = st.slider("Face Y1", 0, image.shape[0], int(image.shape[0] * 0.2))
        face_x2 = st.slider("Face X2", 0, image.shape[1], int(image.shape[1] * 0.7))
        face_y2 = st.slider("Face Y2", 0, image.shape[0], int(image.shape[0] * 0.5))
    
    hand2_coords = None
    if option == "Two-Handed Sign":
        with st.container():
            st.write("**Adjust Hand 2 Bounding Box**")
            hand2_x1 = st.slider("Hand 2 X1", 0, image.shape[1], int(image.shape[1] * 0.4))
            hand2_y1 = st.slider("Hand 2 Y1", 0, image.shape[0], int(image.shape[0] * 0.6))
            hand2_x2 = st.slider("Hand 2 X2", 0, image.shape[1], int(image.shape[1] * 0.8))
            hand2_y2 = st.slider("Hand 2 Y2", 0, image.shape[0], int(image.shape[0] * 0.9))
            hand2_coords = (hand2_x1, hand2_y1, hand2_x2, hand2_y2)
    
    face_coords = (face_x1, face_y1, face_x2, face_y2)
    hand_coords = (hand_x1, hand_y1, hand_x2, hand_y2)
    bboxes = [face_coords, hand_coords]
    if hand2_coords:
        bboxes.append(hand2_coords)
    
    preview_image = draw_bounding_boxes(image, bboxes)
    st.image(preview_image, caption="Updated Bounding Boxes", use_column_width=True)
    
    if st.button("Generate Synthetic Images"):
        synthetic_images, updated_bboxes = generate_synthetic_images(image, bboxes, num_images)
        
        image_files = []
        metadata = {"images": []}
        
        for i, img in enumerate(synthetic_images):
            img_path = f"synthetic_image_{i}.jpg"
            cv2.imwrite(img_path, img)
            image_files.append(img_path)
            metadata["images"].append({"file": img_path, "bboxes": updated_bboxes[i]})
        
        zip_name = "synthetic_images.zip"
        create_zip(image_files, metadata, zip_name)
        st.download_button("Download ZIP", data=open(zip_name, 'rb').read(), file_name=zip_name)
