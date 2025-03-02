import streamlit as st
import pygame
import sys

# Initialize pygame
pygame.init()

# Define screen dimensions
WIDTH, HEIGHT = 400, 400

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 204, 0)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SASL Avatar - Hello")

# Hand representation (a simple rectangle waving)
hand = pygame.Rect(WIDTH // 2 - 25, HEIGHT // 2, 50, 80)
wave_up = False  # Track wave direction

# Function to animate hand waving
def animate_hand():
    global wave_up
    if wave_up:
        hand.y += 10  # Move hand down
    else:
        hand.y -= 10  # Move hand up

    if hand.y <= HEIGHT // 2 - 20:
        wave_up = True
    elif hand.y >= HEIGHT // 2 + 20:
        wave_up = False

# Streamlit UI
st.title("South African Sign Language (SASL) - Avatar Signing 'Hello'")

if st.button("Start Animation"):
    running = True
    while running:
        screen.fill(WHITE)  # Background
        pygame.draw.rect(screen, YELLOW, hand)  # Draw hand

        animate_hand()  # Move hand

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        pygame.display.flip()  # Update screen
        pygame.time.delay(200)  # Slow down animation

st.write("Click the button to see a waving hand sign for 'Hello'!")
