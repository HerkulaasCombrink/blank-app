import streamlit as st
import pandas as pd

def generate_annotation(h, c, n, d1, l1, s1, p1, m1, r1, d2, l2, s2, p2, m2, r2):
    math_annotation = (
        f"({h},{c},{n}) \times \left[ "
        f"\frac{{{d2}}}{{{d1}}} \Bigg| "
        f"\frac{{{l2}}}{{{l1}}} \Bigg| "
        f"\frac{{{s2}}}{{{s1}}} \Bigg| "
        f"\frac{{{p2}}}{{{p1}}} \Bigg| "
        f"\frac{{{m2}}}{{{m1}}} \Bigg| "
        f"\frac{{{r2}}}{{{r1}}} "
        f"\right]"
    )
    
    csv_row = [h, c, n, d1, l1, s1, p1, m1, r1, d2, l2, s2, p2, m2, r2]
    return math_annotation, csv_row

st.title("SASL Annotation Generator")

# Generate annotation and CSV output
math_annotation, csv_row = generate_annotation(1, 0, 1, 1, 1, "B", 1, 4, 1, 2, 0, "0", 0, 0, 0)

# Display results
st.subheader("Mathematical Notation")
st.latex(math_annotation)

st.subheader("CSV Output")
st.write(",".join(map(str, csv_row)))

# Allow CSV Download
df = pd.DataFrame([csv_row], columns=["H", "C", "N", "D1", "L1", "S1", "P1", "M1", "R1", "D2", "L2", "S2", "P2", "M2", "R2"])
st.download_button("Download CSV", df.to_csv(index=False), "annotation.csv", "text/csv")

# Adjustable sliders
st.subheader("Adjust Parameters")
h = st.slider("Handedness (H)", 1, 2, 1)
c = st.slider("Contact (C)", 0, 1, 0)
n = st.slider("Mouthing (N)", 1, 2, 1)
l1 = st.slider("Location (L1)", 0, 10, 1)
p1 = st.slider("Palm Orientation (P1)", 1, 5, 1)
m1 = st.slider("Movement (M1)", 0, 23, 4)
r1 = st.slider("Repetition (R1)", 0, 1, 1)
d2 = st.slider("Non-Dominant Hand (D2)", 0, 2, 2)
l2 = st.slider("Location (L2)", 0, 10, 0)
p2 = st.slider("Palm Orientation (P2)", 0, 5, 0)
m2 = st.slider("Movement (M2)", 0, 23, 0)
r2 = st.slider("Repetition (R2)", 0, 1, 0)
