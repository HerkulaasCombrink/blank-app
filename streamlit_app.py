import streamlit as st
import open3d as o3d
import numpy as np

# Function to create a basic 3D hand model (cube as a simple hand)
def create_hand():
    hand = o3d.geometry.TriangleMesh.create_box(width=0.3, height=0.6, depth=0.1)
    hand.compute_vertex_normals()
    hand.paint_uniform_color([1, 0.8, 0])  # Yellow color
    return hand

# Function to create an animation of waving
def animate_wave():
    hand = create_hand()
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(hand)
    
    for i in range(30):  # Loop to create a waving motion
        angle = 0.2 * np.sin(i * np.pi / 10)  # Oscillating wave motion
        hand.rotate(angle, center=(0, 0, 0))  # Rotate around the center
        vis.update_geometry(hand)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

# Streamlit UI
st.title("SASL 3D Avatar - Signing 'Hello'")

if st.button("Show 3D Hello Sign"):
    animate_wave()
    st.write("3D hand waving animation displayed!")

st.write("This 3D avatar represents a waving hand for the 'Hello' sign.")
