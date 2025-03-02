import streamlit as st
import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
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

# Train model with progress bar
def train_model(uploaded_files):
    images, labels = [], []
    label_map = {}
    
    for uploaded_file in uploaded_files:
        label = uploaded_file.name.split('_')[0]  # Assuming label is in filename
        if label not in label_map:
            label_map[label] = len(label_map)
        
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        images.append(image_array)
        labels.append(label_map[label])
    
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

# Load trained model
def load_model(filename='sign_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Streamlit app
st.title("Sign Language Recognition")
option = st.radio("Choose an action:", ["Train Model", "Detect Sign Using Webcam"])

if option == "Train Model":
    uploaded_files = st.file_uploader("Upload Training Images", type=["jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Train Model") and uploaded_files:
        model, label_map = train_model(uploaded_files)
        save_model(model, label_map)
        st.success("Model trained and saved successfully!")

elif option == "Detect Sign Using Webcam":
    model, label_map = load_model()
    st.write("Webcam Stream (Press 'Start' to detect signs)")
    run = st.checkbox("Start Webcam")
    
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened() and run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam. Make sure it is connected and accessible.")
                break
            
            frame_resized = cv2.resize(frame, (64, 64))
            frame_array = img_to_array(frame_resized) / 255.0
            frame_array = np.expand_dims(frame_array, axis=0)
            prediction = model.predict(frame_array)
            predicted_label = list(label_map.keys())[np.argmax(prediction)]
            
            cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(frame, channels="BGR")
        cap.release()
