import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Hands for Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to process video and detect hands
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    frames = []
    translations = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            translations.append("Detected SASL Gesture")  # Placeholder for sign translation
        
        frames.append(frame)
    
    cap.release()
    
    return frames, translations

# Function to save processed video
def save_processed_video(frames):
    height, width, _ = frames[0].shape
    output_path = "processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

    for frame in frames:
        out.write(frame)
    
    out.release()
    return output_path

# Streamlit UI
st.title("SASL Video Translation App")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.video(video_path)

    if st.button("Process Video"):
        with st.spinner("Processing video..."):
            frames, translations = process_video(video_path)
            processed_video_path = save_processed_video(frames)
        
        st.success("Video processed successfully!")
        st.video(processed_video_path)
        
        st.subheader("Translation:")
        for i, translation in enumerate(translations):
            st.write(f"Frame {i}: {translation}")

