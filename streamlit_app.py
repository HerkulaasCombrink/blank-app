import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# Function to create a simple 3D hand using Plotly
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
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=palm_x, y=rotated_y, z=rotated_z, mode='lines', line=dict(color='orange', width=5)))
    
    for finger in fingers:
        finger_y = np.cos(angle) * np.array(finger[1]) - np.sin(angle) * np.array(finger[2])
        finger_z = np.sin(angle) * np.array(finger[1]) + np.cos(angle) * np.array(finger[2])
        fig.add_trace(go.Scatter3d(x=finger[0], y=finger_y, z=finger_z, mode='lines', line=dict(color='brown', width=4)))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# Streamlit UI
st.title("SASL 3D Avatar - Signing 'Hello'")

if st.button("Show 3D Hello Sign"):
    for i in range(20):  # Animate the waving motion
        angle = 0.3 * np.sin(i * np.pi / 5)
        fig = create_hand(angle)
    
    st.plotly_chart(fig)  # Show the final frame in one view

st.write("This 3D avatar represents a waving hand for the 'Hello' sign.")
