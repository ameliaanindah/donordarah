import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
st.markdown("""
<div class="card">
    <h1>🩸 Prediksi Kelayakan Donor Darah</h1>
    <p>
        Masukkan data kesehatan pendonor untuk mengetahui
        status kelayakan donor darah secara cepat.
    </p>
</div>
""", unsafe_allow_html=True)

# ===========================
# LOAD & TRAIN MODEL
# ===========================
@st.cache_data
def load_data():
    return pd.read_csv("data_bersih.csv")

df = load_data()

X = df.drop(["status", "jenis_kelamin"], axis=1)
y = df["status"]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ===========================
# INPUT USER
# ===========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Form Data Pendonor")

berat_badan = st.number_input("Berat Badan (kg)", 30.0, 200.0)
suhu = st.number_input("Suhu Tubuh (°C)", 35.0, 40.0)
nadi = st.number_input("Nadi (x/menit)", 40.0, 150.0)
hb = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0)
hct = st.number_input("Hematokrit (%)", 10.0, 60.0)
umur = st.number_input("Umur (tahun)", 17.0, 65.0)
sistole = st.number_input("Tensi Sistole", 80.0, 200.0)
diastole = st.number_input("Tensi Diastole", 40.0, 120.0)

st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# PREDIKSI
# ===========================
if st.button("Cek Kelayakan Donor"):
    new_data = pd.DataFrame(
        [[berat_badan, suhu, nadi, hb, hct, umur, sistole, diastole]],
        columns=X.columns
    )

    pred = rf.predict(new_data)[0]
    proba = rf.predict_proba(new_data)[0]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Hasil Pemeriksaan")

    if pred == 0:
        st.markdown("<div class='layak'>LAYAK DONOR DARAH</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='tidak-layak'>TIDAK LAYAK DONOR DARAH</div>", unsafe_allow_html=True)

    st.write(f"Probabilitas Layak: **{proba[0]*100:.2f}%**")
    st.write(f"Probabilitas Tidak Layak: **{proba[1]*100:.2f}%**")

    st.markdown("</div>", unsafe_allow_html=True)
