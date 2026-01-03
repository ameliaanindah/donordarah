import streamlit as st
import pandas as pd
import pickle

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Prediksi Donor Darah",
    layout="centered"
)

# ===========================
# LOAD CSS
# ===========================
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===========================
# HEADER
# ===========================
st.title("🩸 Prediksi Kelayakan Donor Darah")
st.write(
    "Masukkan data kesehatan pendonor untuk mengetahui "
    "status kelayakan donor darah."
)

# ===========================
# LOAD MODEL
# ===========================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
# ===========================
# INPUT USER
# ===========================
st.subheader("Form Data Pendonor")

berat_badan = st.number_input(
    "Berat Badan (kg)",
    min_value=30,
    max_value=200,
    step=1,
    format="%d"
)

suhu = st.number_input(
    "Suhu Tubuh (°C)",
    min_value=35.0,
    max_value=40.0,
    step=0.1
)

nadi = st.number_input(
    "Nadi (x/menit)",
    min_value=40,
    max_value=150,
    step=1,
    format="%d"
)

hb = st.number_input(
    "Hemoglobin (g/dL)",
    min_value=5.0,
    max_value=20.0,
    step=0.1
)

hct = st.number_input(
    "Hematokrit (%)",
    min_value=10.0,
    max_value=60.0,
    step=0.1
)

umur = st.number_input(
    "Umur (tahun)",
    min_value=17,
    max_value=65,
    step=1,
    format="%d"
)

sistole = st.number_input(
    "Tensi Sistole",
    min_value=80,
    max_value=200,
    step=1,
    format="%d"
)

diastole = st.number_input(
    "Tensi Diastole",
    min_value=40,
    max_value=120,
    step=1,
    format="%d"
)

# ===========================
# PREDIKSI
# ===========================
if st.button("Cek Kelayakan Donor"):
    new_data = pd.DataFrame(
        [[berat_badan, suhu, nadi, hb, hct, umur, sistole, diastole]],
        columns=[
            "berat_badan", "suhu", "nadi",
            "hb", "hct", "umur",
            "sistole", "diastole"
        ]
    )

    pred = model.predict(new_data)[0]
    proba = model.predict_proba(new_data)[0]

    st.subheader("Hasil Pemeriksaan")

    if pred == 0:
        st.success("LAYAK DONOR DARAH")
    else:
        st.error("TIDAK LAYAK DONOR DARAH")

    st.write(f"Probabilitas Layak: **{proba[0]*100:.2f}%**")
    st.write(f"Probabilitas Tidak Layak: **{proba[1]*100:.2f}%**")
