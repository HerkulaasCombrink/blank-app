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
import mediapipe.python.solutions as mp_solutions  # Use MediaPipe without OpenCV

# Function to create a simple 3D hand using Matplotlib
def create_hand(angle):
    palm_x = [-0.2, 0.2, 0.2, -0.2, -0.2]
    palm_y = [0, 0, 0.4, 0.4, 0]
    palm_z = [1, 1, 1, 1, 1]

    fingers = [
        ([-0.15, -0.15], [0.4, 0.6], [1, 1]),
        ([-0.05, -0.05], [0.4, 0.7], [1, 1]),
        ([0.05, 0.05], [0.4, 0.75], [1, 1]),
        ([0.15, 0.15], [0.4, 0.65], [1, 1])
    ]
    
    # Rotate hand slightly
    rotated_y = np.cos(angle) * np.array(palm_y) - np.sin(angle) * np.array(palm_z)
    rotated_z = np.sin(angle) * np.array(palm_y) + np.cos(angle) * np.array(palm_z)
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': '3d'})
    ax.plot(palm_x, rotated_y, rotated_z, color='orange', linewidth=3)
    
    for finger in fingers:
        finger_y = np.cos(angle) * np.array(finger[1]) - np.sin(angle) * np.array(finger[2])
        finger_z = np.sin(angle) * np.array(finger[1]) + np.cos(angle) * np.array(finger[2])
        ax.plot(finger[0], finger_y, finger_z, color='brown', linewidth=2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("SASL 'Hello' Sign")
    
    return fig

# Generate a looping GIF of the animation using Matplotlib
def generate_gif():
    frames = []
    for i in range(20):  # Animate the waving motion
        angle = 0.3 * np.sin(i * np.pi / 5)
        fig = create_hand(angle)
        
        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        img = Image.open(buf)
        frames.append(img)
        plt.close(fig)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
        imageio.mimsave(tmpfile.name, frames, duration=0.1, loop=0)  # Loop the GIF indefinitely
        return tmpfile.name

# Streamlit UI
st.title("SASL 3D Avatar - Signing 'Hello'")

if st.button("Generate & Show GIF"):
    gif_path = generate_gif()
    st.image(gif_path)
    st.download_button(label="Download GIF", data=open(gif_path, "rb").read(), file_name="hello_sign.gif", mime="image/gif")

st.write("This 3D avatar represents a waving hand for the 'Hello' sign.")
