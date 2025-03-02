from manim import *
import streamlit as st
import os

temp_video_path = "hello_sign.mp4"

def create_hello_animation():
    class HelloSign(Scene):
        def construct(self):
            # Create a waving hand using circles and rectangles
            hand = SVGMobject("hand.svg").scale(1.5).set_color(YELLOW)
            hand.move_to(RIGHT * 2)
            
            text = Text("Hello!").scale(1.5).move_to(UP * 2.5)
            
            # Create a waving motion
            self.play(Write(text))
            for _ in range(3):
                self.play(hand.animate.rotate(PI / 6), run_time=0.3)
                self.play(hand.animate.rotate(-PI / 6), run_time=0.3)
            
            self.wait(1)
    
    HelloSign().render(file_name=temp_video_path)

# Check if the animation exists, otherwise generate it
if not os.path.exists(temp_video_path):
    create_hello_animation()

# Streamlit UI
st.title("SASL Avatar: Signing Hello")

if st.button("Show 'Hello' Sign Animation"):
    st.video(temp_video_path)
