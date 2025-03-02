import streamlit as st
import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from io import BytesIO
from PIL import Image

# Define model architecture
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output neurons match number of classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to save model
def save_model(model, label_map, filename='sign_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((model, label_map), f)
    return filename

# Function to load model
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Automatically detect red bounding boxes
def detect_red_bounding_box(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, x + w, y + h)
    return None

# Extract region of interest (ROI) from image
def extract_roi(image):
    bbox = detect_red_bounding_box(image)
    if bbox:
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (64, 64)) / 255.0
        return roi_resized
    return None

# Data augmentation for training images
def augment_image(image):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )
    image = np.expand_dims(image, axis=0)
    return datagen.flow(image, batch_size=1)[0][0]

# Train model
def train_model(uploaded_files):
    images, labels = [], []
    label_map = {'No Sign': 0}  # Include 'No Sign' class
    
    for uploaded_file in uploaded_files:
        label = uploaded_file.name.split('_')[0]  # Assuming label is in filename
        if label not in label_map:
            label_map[label] = len(label_map)
        
        image = np.array(Image.open(uploaded_file).convert('RGB'))
        roi = extract_roi(image)
        if roi is not None:
            augmented_images = [roi] + [augment_image(roi) for _ in range(5)]  # Augment data
            images.extend(augmented_images)
            labels.extend([label_map[label]] * len(augmented_images))
    
    # Add blank images for 'No Sign' class
    for _ in range(len(images) // len(label_map)):
        blank_image = np.zeros((64, 64, 3))
        images.append(blank_image)
        labels.append(label_map['No Sign'])
    
    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_map))
    
    model = create_model(num_classes=len(label_map))
    
    progress_bar = st.progress(0)
    class TrainingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / 10)
    
    model.fit(images, labels, epochs=10, batch_size=8, validation_split=0.2, callbacks=[TrainingCallback()])
    return model, label_map
