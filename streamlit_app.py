import streamlit as st
import tempfile
import numpy as np
import moviepy.editor as mp
import imageio
import mediapipe as mp_hands
from PIL import Image
import io

# Initialize MediaPipe Hands
mp_hands_module = mp_hands.solutions.hands
mp_drawing = mp_hands.solutions.drawing_utils

# Function to process video and detect hand movements
def process_video(video_path):
    clip = mp.VideoFileClip(video_path)
    hands = mp_hands_module.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    translations = []
    frames = []

    for frame in clip.iter_frames(fps=10, dtype="uint8"):
        image = Image.fromarray(frame)
        frame_rgb = np.array(image)

        # Process frame with MediaPipe
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands_module.HAND_CONNECTIONS)
            translations.append("Detected SASL Gesture")  # Placeholder for real translation

        frames.append(frame_rgb)

    return frames, translations

# Function to save processed video
def save_processed_video(frames, original_video_path):
    clip = mp.VideoFileClip(original_video_path)
    output_path = "processed_video.mp4"
    video_writer = imageio.get_writer(output_path, fps=10)

    for frame in frames:
        video_writer.append_data(frame)

    video_writer.close()
    return output_path

# Streamlit UI
st.title("SASL Video Translation Without OpenCV")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.video(video_path)

    if st.button("Process Video"):
        with st.spinner("Processing video..."):
            frames, translations = process_video(video_path)
            processed_video_path = save_processed_video(frames, video_path)

        st.success("Video processed successfully!")
        st.video(processed_video_path)

        st.subheader("Translation:")
        for i, translation in enumerate(translations):
            st.write(f"Frame {i}: {translation}")
