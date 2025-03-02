import streamlit as st
import os

# Sample dictionary mapping words to SASL sign images or videos (update paths as needed)
SASL_SIGNS = {
    "hello": "signs/hello.gif",
    "thank you": "signs/thank_you.gif",
    "yes": "signs/yes.gif",
    "no": "signs/no.gif",
    "please": "signs/please.gif",
    "help": "signs/help.gif",
}

# Streamlit UI
st.title("South African Sign Language (SASL) Text-to-Sign")

# User input
text_input = st.text_area("Enter text to translate to SASL:", "")

if st.button("Translate"):
    if text_input:
        words = text_input.lower().split()  # Basic tokenization
        found_signs = []
        
        # Search for signs in dictionary
        for word in words:
            if word in SASL_SIGNS:
                found_signs.append((word, SASL_SIGNS[word]))

        # Display results
        if found_signs:
            st.subheader("SASL Translation:")
            for word, sign_path in found_signs:
                st.write(f"**{word.capitalize()}**")
                st.image(sign_path, use_container_width=True)  # Updated parameter
        else:
            st.warning("No matching SASL signs found. Try a different input.")
    else:
        st.warning("Please enter text to translate.")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter text in the box.
2. Click "Translate" to convert to SASL.
3. Matching SASL signs will be displayed.
""")
