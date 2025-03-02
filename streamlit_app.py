import streamlit as st
import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
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
            # Run YOLO to detect potential sign areas
            detections = yolo_model(frame)
            
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection.xyxy[0].cpu().numpy()
                if conf < 0.7:  # Ignore low-confidence detections
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
    option = st.radio("Choose an action:", ["Upload Pickle File and Process Video"])
    
    if option == "Upload Pickle File and Process Video":
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
