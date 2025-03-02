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

# Train model with progress bar using automatically detected ROIs
def train_model(uploaded_files):
    images, labels = [], []
    label_map = {}
    
    for uploaded_file in uploaded_files:
        label = uploaded_file.name.split('_')[0]  # Assuming label is in filename
        if label not in label_map:
            label_map[label] = len(label_map)
        
        image = np.array(Image.open(uploaded_file).convert('RGB'))
        roi = extract_roi(image)
        if roi is not None:
            images.append(roi)
            labels.append(label_map[label])
    
    if not images:
        st.error("No valid bounding boxes detected in the images.")
        return None, None
    
    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_map))
    
    model = create_model(num_classes=len(label_map))
    
    progress_bar = st.progress(0)
    class TrainingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / 10)
    
    model.fit(images, labels, epochs=10, batch_size=8, validation_split=0.2, callbacks=[TrainingCallback()])
    return model, label_map

# Save trained model
def save_model(model, label_map, filename='sign_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((model, label_map), f)
    return filename

# Load trained model
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Process video and detect signs using automatically detected bounding boxes
def process_video(video_file, model, label_map):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    detected_signs = []
    
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
                predicted_label = list(label_map.keys())[np.argmax(prediction)]
                detected_signs.append({"Second": frame_count // fps, "Detected Sign": predicted_label})
        
        frame_count += 1
    
    cap.release()
    return pd.DataFrame(detected_signs)

# Streamlit app
st.title("Sign Language Recognition")
option = st.radio("Choose an action:", ["Train Model", "Upload Pickle File and Process Video"])

if option == "Train Model":
    uploaded_files = st.file_uploader("Upload Training Images", type=["jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Train Model") and uploaded_files:
        model, label_map = train_model(uploaded_files)
        if model:
            model_filename = save_model(model, label_map)
            st.success("Model trained and saved successfully!")
            with open(model_filename, "rb") as f:
                st.download_button("Download Model", f, file_name=model_filename)

elif option == "Upload Pickle File and Process Video":
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
