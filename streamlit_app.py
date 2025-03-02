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

# Process video and detect signs using automatically detected bounding boxes
def process_video(video_file, model, label_map):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    detected_signs = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % fps == 0:  # Process 1 frame per second
            roi = extract_roi(frame)
            if roi is not None:
                frame_array = np.expand_dims(roi, axis=0)
                prediction = model.predict(frame_array)
                predicted_index = np.argmax(prediction)
                
                if prediction[0][predicted_index] >= 0.7:  # Confidence threshold
                    predicted_label = list(label_map.keys())[predicted_index]
                else:
                    predicted_label = "Not Detected"
                
                detected_signs.append({"Second": frame_count // fps, "Detected Sign": predicted_label})
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    return pd.DataFrame(detected_signs)

# Streamlit app
st.title("Sign Language Recognition")
option = st.radio("Choose an action:", ["Upload Pickle File and Process Video"])

if option == "Upload Pickle File and Process Video":
    uploaded_pkl = st.file_uploader("Upload Trained Model (Pickle File)", type=["pkl"])
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4"])
    
    if uploaded_pkl and uploaded_video:
        with open("temp_model.pkl", "wb") as f:
            f.write(uploaded_pkl.getbuffer())
        model, label_map = load_model("temp_model.pkl")
        st.success("Model loaded successfully!")
        
        if st.button("Process Video"):
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            st.success("Video uploaded successfully! Processing...")
            results_df = process_video("temp_video.mp4", model, label_map)
            
            st.write("Detected Signs in Video:")
            st.dataframe(results_df)
            
            csv_filename = "detected_signs.csv"
            results_df.to_csv(csv_filename, index=False)
            st.download_button("Download CSV", data=open(csv_filename, "rb").read(), file_name=csv_filename)
