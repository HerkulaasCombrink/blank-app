import streamlit as st
import os

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dictionary mapping words to SASL sign images/videos (update paths accordingly)
SASL_SIGNS = {
    "hello": os.path.join(BASE_DIR, "signs/hello.gif"),
    "thank you": os.path.join(BASE_DIR, "signs/thank_you.gif"),
    "yes": os.path.join(BASE_DIR, "signs/yes.gif"),
    "no": os.path.join(BASE_DIR, "signs/no.gif"),
    "please": os.path.join(BASE_DIR, "signs/please.gif"),
    "help": os.path.join(BASE_DIR, "signs/help.gif"),
}

# Streamlit UI
st.title("South African Sign Language (SASL) Text-to-Sign")
st.write("Enter text, and the app will display corresponding SASL signs.")

# User input
text_input = st.text_area("Enter text to translate to SASL:", "")

if st.button("Translate"):
    if text_input:
        words = text_input.lower().split()  # Basic tokenization
        found_signs = []

        # Search for signs in dictionary
        for word in words:
            if word in SASL_SIGNS:
                sign_path = SASL_SIGNS[word]
                
                # Verify file exists before displaying
                if os.path.exists(sign_path):
                    found_signs.append((word, sign_path))
                else:
                    st.warning(f"Sign file missing for: {word}")

        # Display results
        if found_signs:
            st.subheader("SASL Translation:")
            for word, sign_path in found_signs:
                st.write(f"**{word.capitalize()}**")
                
                # Display GIFs/Videos or images correctly
                if sign_path.endswith((".mp4", ".gif")):
                    st.video(sign_path)  # Use for GIFs/Videos
                else:
                    st.image(sign_path, use_container_width=True)
        else:
            st.warning("No matching SASL signs found. Try different input.")
    else:
        st.warning("Please enter text to translate.")

# Sidebar Instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter text in the box.
2. Click "Translate" to convert to SASL.
3. Matching SASL signs will be displayed.
4. Ensure that sign media files are placed in the `signs/` folder.
""")

# Debugging: Show available sign files
if st.sidebar.checkbox("Show available sign files"):
    available_files = [f for f in os.listdir(os.path.join(BASE_DIR, "signs")) if f.endswith((".gif", ".mp4", ".jpg", ".png"))]
    st.sidebar.write(available_files)
