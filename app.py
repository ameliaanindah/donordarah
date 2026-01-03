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

berat_badan = st.number_input("Berat Badan (kg)", 30.0, 200.0)
suhu = st.number_input("Suhu Tubuh (°C)", 35.0, 40.0)
nadi = st.number_input("Nadi (x/menit)", 40.0, 150.0)
hb = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0)
hct = st.number_input("Hematokrit (%)", 10.0, 60.0)
umur = st.number_input("Umur (tahun)", 17.0, 65.0)
sistole = st.number_input("Tensi Sistole", 80.0, 200.0)
diastole = st.number_input("Tensi Diastole", 40.0, 120.0)

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
