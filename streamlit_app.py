import streamlit as st
import pandas as pd

def generate_annotation(h, c, n, d1, l1, s1, p1, m1, r1, d2, l2, s2, p2, m2, r2):
    math_annotation = (
        f"({h},{c},{n}) × [ "
        f"{d2}/{d1} | "
        f"{l2}/{l1} | "
        f"{s2}/{s1} | "
        f"{p2}/{p1} | "
        f"{m2}/{m1} | "
        f"{r2}/{r1} "
        f"]"
    )
    
    csv_row = [h, c, n, d1, l1, s1, p1, m1, r1, d2, l2, s2, p2, m2, r2]
    return math_annotation, csv_row

st.title("SASL Annotation Generator")

# Sidebar inputs
st.sidebar.header("Global Features")
h = st.sidebar.selectbox("Handedness (H)", [1, 2], index=0)
c = st.sidebar.selectbox("Contact (C)", [0, 1], index=0)
n = st.sidebar.selectbox("Mouthing (N)", [1, 2], index=0)

st.sidebar.header("Dominant Hand")
d1 = 1  # Dominant hand always 1
l1 = st.sidebar.selectbox("Location (L1)", list(range(11)), index=1)
s1 = st.sidebar.text_input("Handshape (S1)", "B")
p1 = st.sidebar.selectbox("Palm Orientation (P1)", list(range(1, 6)), index=0)
m1 = st.sidebar.selectbox("Movement (M1)", list(range(24)), index=0)
r1 = st.sidebar.selectbox("Repetition (R1)", [0, 1], index=0)

st.sidebar.header("Non-Dominant Hand")
d2 = st.sidebar.selectbox("Non-Dominant Hand (D2)", [0, 2], index=0)
l2 = st.sidebar.selectbox("Location (L2)", list(range(11)), index=0)
s2 = st.sidebar.text_input("Handshape (S2)", "0")
p2 = st.sidebar.selectbox("Palm Orientation (P2)", list(range(6)), index=0)
m2 = st.sidebar.selectbox("Movement (M2)", list(range(24)), index=0)
r2 = st.sidebar.selectbox("Repetition (R2)", [0, 1], index=0)

# Generate annotation and CSV output
math_annotation, csv_row = generate_annotation(h, c, n, d1, l1, s1, p1, m1, r1, d2, l2, s2, p2, m2, r2)

# Display results
st.subheader("Mathematical Notation")
st.latex(math_annotation)

st.subheader("CSV Output")
st.write(",".join(map(str, csv_row)))

# Allow CSV Download
df = pd.DataFrame([csv_row], columns=["H", "C", "N", "D1", "L1", "S1", "P1", "M1", "R1", "D2", "L2", "S2", "P2", "M2", "R2"])
st.download_button("Download CSV", df.to_csv(index=False), "annotation.csv", "text/csv")
