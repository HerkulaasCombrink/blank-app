import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Function to create a simple 3D hand
def create_hand(ax, angle):
    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.set_title("SASL 'Hello' Sign - Waving Hand")

    # Create a simple hand using a rectangle (palm) and fingers (lines)
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

    ax.plot(palm_x, rotated_y, rotated_z, color='orange', linewidth=3)

    for finger in fingers:
        finger_y = np.cos(angle) * np.array(finger[1]) - np.sin(angle) * np.array(finger[2])
        finger_z = np.sin(angle) * np.array(finger[1]) + np.cos(angle) * np.array(finger[2])
        ax.plot(finger[0], finger_y, finger_z, color='brown', linewidth=2)

# Streamlit UI
st.title("SASL 3D Avatar - Signing 'Hello'")

if st.button("Show 3D Hello Sign"):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(20):  # Animate the waving motion
        angle = 0.3 * np.sin(i * np.pi / 5)
        create_hand(ax, angle)
        st.pyplot(fig)
        time.sleep(0.1)

st.write("This 3D avatar represents a waving hand for the 'Hello' sign.")
