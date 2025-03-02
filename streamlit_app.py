import streamlit as st
import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# Define model architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # Adjust output neurons based on classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(data_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    model = create_model()
    model.fit(train_generator, validation_data=val_generator, epochs=10)
    return model

# Save trained model
def save_model(model, filename='sign_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Load trained model
def load_model(filename='sign_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Streamlit app
st.title("Sign Language Recognition")
option = st.radio("Choose an action:", ["Train Model", "Detect Sign Using Webcam"])

if option == "Train Model":
    data_dir = st.text_input("Enter dataset folder path:")
    if st.button("Train Model") and os.path.exists(data_dir):
        model = train_model(data_dir)
        save_model(model)
        st.success("Model trained and saved successfully!")

elif option == "Detect Sign Using Webcam":
    model = load_model()
    st.write("Webcam Stream (Press 'Start' to detect signs)")
    run = st.checkbox("Start Webcam")
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (64, 64))
        frame_array = img_to_array(frame_resized) / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)
        prediction = model.predict(frame_array)
        predicted_label = np.argmax(prediction)
        st.image(frame, caption=f"Predicted Sign: {predicted_label}", channels="BGR")
    cap.release()
