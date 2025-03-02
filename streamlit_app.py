import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import imageio
import io
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mediapipe as mp
import tensorflow as tf

# Load pre-trained AI model for SASL classification
@st.cache(allow_output_mutation=True)
def load_sasl_model():
    model = tf.keras.models.load_model("sasl_sign_model.h5")  # Load your trained model
    return model

sasl_model = load_sasl_model()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# SASL dictionary for sign translation
SASL_GESTURE_MAP = {
    0: "Hello",
    1: "Goodbye",
    2: "Thank You",
    3: "Yes",
    4: "No",
}

# Function to create a 3D avatar hand visualization
def create_hand(sign_text):
    fig = go.Figure()
    
    # Adjust hand position based on sign
    if sign_text == "Hello":
        palm_y = [0, 0.3, 0.6, 0.3, 0]
    elif sign_text == "Goodbye":
        palm_y = [0, 0.4, 0.4, 0, 0]
    else:
        palm_y = [0, 0.2, 0.5, 0.2, 0]

    palm_x = [-0.2, 0.2, 0.2, -0.2, -0.2]
    palm_z = [1, 1, 1, 1, 1]

    fig.add_trace(go.Scatter3d(x=palm_x, y=palm_y, z=palm_z, mode='lines', line=dict(color='orange', width=5)))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

# Function to classify SASL gestures
def classify_sasl_sign(hand_landmarks):
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    keypoints = keypoints.reshape(1, -1)  # Reshape for model input
    prediction = sasl_model.predict(keypoints)
    sign_index = np.argmax(prediction)
    return SASL_GESTURE_MAP.get(sign_index, "Unknown Sign")

# Streamlit UI
st.title("AI-Powered SASL Translator")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.video(video_path)

    if st.button("Translate Video to SASL"):
        with st.spinner("Processing video..."):
            cap = imageio.get_reader(video_path)
            detected_signs = []

            for frame in cap:
                image = Image.fromarray(frame)
                frame_rgb = np.array(image)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        sign_text = classify_sasl_sign(hand_landmarks)
                        detected_signs.append(sign_text)

            detected_signs = list(set(detected_signs))  # Remove duplicates
            st.success(f"Detected SASL Signs: {', '.join(detected_signs)}")

            # Display 3D Avatar of First Detected Sign
            if detected_signs:
                sign_to_display = detected_signs[0]
                avatar_fig = create_hand(sign_to_display)
                st.plotly_chart(avatar_fig)

st.write("This app translates uploaded videos into SASL and displays a 3D avatar representation.")
