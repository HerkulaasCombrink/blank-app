import streamlit as st
import pandas as pd

def generate_annotation(h, c, n, variation, l1, s1, p1, m1, r1, l2, s2, p2, m2, r2):
    math_annotation = (
        f"S = ({h}, {c}, {n}) \\times \\left[ "
        f"\\frac{{{l2}^2}}{{{l1}_1}} \\Bigg| "
        f"\\frac{{{s2}^2}}{{{s1}_1}} \\Bigg| "
        f"\\frac{{{p2}^2}}{{{p1}_1}} \\Bigg| "
        f"\\frac{{{m2}^2}}{{{m1}_1}} \\Bigg| "
        f"\\frac{{{r2}^2}}{{{r1}_1}} "
        f"\\right]"
    )
    
    csv_row = [variation, h, c, n, l1, s1, p1, m1, r1, l2, s2, p2, m2, r2]
    return math_annotation, csv_row

st.title("SASL Annotation Generator")

# Variation input
variation = st.text_input("Sign Variation Label", "Variation 1")

# Layout for global parameters
st.subheader("Global Parameters")
col_global1, col_global2, col_global3 = st.columns(3)
with col_global1:
    h = st.selectbox("Handedness (H)", [1, 2], index=0)
with col_global2:
    c = st.selectbox("Contact (C)", [0, 1], index=0)
with col_global3:
    n = st.selectbox("Mouthing (N)", [1, 2], index=0)

# Layout for dominant and non-dominant hand
st.subheader("Dominant and Non-Dominant Hand Parameters")
col1, col2 = st.columns(2)
with col1:
    st.text("Dominant Hand")
    l1 = st.slider("Location (L1)", 0, 10, 1)
    s1 = st.text_input("Handshape (S1)", "B")
    p1 = st.slider("Palm Orientation (P1)", 1, 5, 1)
    m1 = st.slider("Movement (M1)", 0, 23, 4)
    r1 = st.slider("Repetition (R1)", 0, 1, 1)

with col2:
    st.text("Non-Dominant Hand")
    if h == 2:
        l2 = st.slider("Location (L2)", 0, 10, 0)
        s2 = st.text_input("Handshape (S2)", "0")
        p2 = st.slider("Palm Orientation (P2)", 0, 5, 0)
        m2 = st.slider("Movement (M2)", 0, 23, 0)
        r2 = st.slider("Repetition (R2)", 0, 1, 0)
    else:
        l2, s2, p2, m2, r2 = 0, "0", 0, 0, 0  # Defaults for non-dominant hand when one-handed

# Calculate button
if st.button("Calculate Annotation"):
    math_annotation, csv_row = generate_annotation(h, c, n, variation, l1, s1, p1, m1, r1, l2, s2, p2, m2, r2)
    
    # Display results
    st.subheader("Mathematical Notation")
    st.latex(math_annotation)
    
    st.subheader("CSV Output")
    st.write(",".join(map(str, csv_row)))
    
    # Allow CSV Download
    df = pd.DataFrame([csv_row], columns=["Variation", "H", "C", "N", "L1", "S1", "P1", "M1", "R1", "L2", "S2", "P2", "M2", "R2"])
    st.download_button("Download CSV", df.to_csv(index=False), "annotation.csv", "text/csv")
