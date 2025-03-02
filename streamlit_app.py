import streamlit as st
import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load YOLO model
from ultralytics import YOLO

# Load pre-trained YOLO model for object detection
def load_yolo_model(model_path):
    return YOLO(model_path)

# Train the classification model
def train_model(uploaded_files):
    images, labels = [], []
    label_map = {}
    
    for uploaded_file in uploaded_files:
        label = uploaded_file.name.split('_')[0]  # Extract label from filename
        if label not in label_map:
            label_map[label] = len(label_map)
        
        image = np.array(Image.open(uploaded_file).convert('RGB'))
        image_resized = cv2.resize(image, (64, 64)) / 255.0
        images.append(image_resized)
        labels.append(label_map[label])
    
    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_map))
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_map), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=10, batch_size=8, validation_split=0.2)
    return model, label_map

# Train YOLO model on uploaded images
def train_yolo(uploaded_files):
    yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 base model
    data_dir = "yolo_training_data"
    os.makedirs(data_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    yolo_model.train(data=data_dir, epochs=10, imgsz=640)
    return "runs/train/exp/weights/best.pt"

# Process video and detect trained sign patterns
def process_video(video_file, model, label_map, yolo_model):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = 5  # Process every 5th frame
    detected_signs = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            detections = yolo_model(frame)
            
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection.xyxy[0].cpu().numpy()
                if conf < 0.7:
                    continue
                
                roi = frame[int(y1):int(y2), int(x1):int(x2)]
                roi_resized = cv2.resize(roi, (64, 64)) / 255.0
                frame_array = np.expand_dims(roi_resized, axis=0)
                
                prediction = model.predict(frame_array)
                predicted_index = np.argmax(prediction)
                confidence = prediction[0][predicted_index]
                
                if confidence >= 0.7:
                    predicted_label = list(label_map.keys())[predicted_index]
                else:
                    predicted_label = "Not Detected"
                
                detected_signs.append({"Timestamp (s)": frame_count // fps, "Detected Sign": predicted_label, "Confidence": confidence})
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    return pd.DataFrame(detected_signs)

# Streamlit App
def main():
    st.title("Sign Language Recognition App")
    option = st.radio("Choose an action:", ["Train Model", "Upload Pickle File and Process Video"])
    
    if option == "Train Model":
        uploaded_files = st.file_uploader("Upload Training Images", type=["jpg", "jpeg"], accept_multiple_files=True)
        if st.button("Train Model") and uploaded_files:
            model, label_map = train_model(uploaded_files)
            yolo_weights = train_yolo(uploaded_files)
            
            model_filename = "sign_model.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump((model, label_map), f)
            
            st.success("Model trained successfully! You can now download the trained models.")
            st.download_button("Download Classification Model", open(model_filename, "rb"), file_name=model_filename)
            st.download_button("Download YOLO Weights", open(yolo_weights, "rb"), file_name="sign_detector.pt")
    
    elif option == "Upload Pickle File and Process Video":
        uploaded_pkl = st.file_uploader("Upload Trained Model (Pickle File)", type=["pkl"])
        uploaded_video = st.file_uploader("Upload Video File", type=["mp4"])
        uploaded_yolo = st.file_uploader("Upload YOLO Model (Weights)", type=["pt"])
        
        if uploaded_pkl and uploaded_video and uploaded_yolo:
            with open("temp_model.pkl", "wb") as f:
                f.write(uploaded_pkl.getbuffer())
            model, label_map = pickle.load(open("temp_model.pkl", "rb"))
            
            with open("temp_yolo.pt", "wb") as f:
                f.write(uploaded_yolo.getbuffer())
            yolo_model = load_yolo_model("temp_yolo.pt")
            
            st.success("Models loaded successfully!")
            
            if st.button("Process Video"):
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                st.success("Video uploaded successfully! Processing...")
                results_df = process_video("temp_video.mp4", model, label_map, yolo_model)
                
                st.write("Detected Signs in Video:")
                st.dataframe(results_df)
                
                csv_filename = "detected_signs.csv"
                results_df.to_csv(csv_filename, index=False)
                st.download_button("Download CSV", data=open(csv_filename, "rb").read(), file_name=csv_filename)

if __name__ == "__main__":
    main()
 
