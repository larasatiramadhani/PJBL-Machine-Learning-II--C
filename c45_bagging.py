import streamlit as st
import numpy as np
import pickle

st.title("📌 Prediksi Kelangsungan Hidup Pasien Sirosis Hati")
st.write("""
Aplikasi ini menggunakan **Bagging C4.5** untuk memprediksi kondisi pasien berdasarkan data klinis dan hasil laboratorium.
Silakan isi data pasien dengan benar untuk mendapatkan prediksi.
""")

# Load model Bagging
with open("c45_bagging.pkl", "rb") as f:
    models = pickle.load(f)

def predict_bagging(input_data):
    preds = [m.predict([input_data])[0] for m in models]
    vals, counts = np.unique(preds, return_counts=True)
    return vals[np.argmax(counts)]

def encode(drug, sex, asc, hep, spi, ede):
    # encoding sesuai training
    drug_map = {"D-penicillamine": 0, "Placebo": 1}
    sex_map = {"Perempuan": 0, "Laki-Laki": 1}
    yn_map = {"No": 0, "Yes": 1}
    ede_map = {"No": 0, "Some": 1, "Yes": 2}  # Some = edema ringan/resolved
    return [drug_map[drug], sex_map[sex], yn_map[asc], yn_map[hep], yn_map[spi], ede_map[ede]]

# -----------------------------
# Input Form
# -----------------------------
st.subheader("🧍 Data Klinis Pasien")
col1, col2 = st.columns(2)

with col1:
    N_Days = st.number_input("Lama hari sejak registrasi (N_Days)", min_value=1, step=1)
    Age = st.number_input("Usia Pasien (hari)", min_value=0, step=1)
    Sex = st.selectbox("Jenis Kelamin (Sex)", ["-- Pilih --", "Perempuan", "Laki-Laki"])
    Drug = st.selectbox("Jenis Obat (Drug)", ["-- Pilih --", "D-penicillamine", "Placebo"], help="Jenis obat yang diterima pasien")

with col2:
    Ascites = st.selectbox("Ascites", ["-- Pilih --", "No", "Yes"], help="No = tidak ada, Yes = ada")
    Hepatomegaly = st.selectbox("Hepatomegaly", ["-- Pilih --", "No", "Yes"], help="No = tidak ada, Yes = ada")
    Spiders = st.selectbox("Spiders", ["-- Pilih --", "No", "Yes"], help="No = tidak ada, Yes = ada")
    Edema = st.selectbox("Edema", ["-- Pilih --", "No", "Some", "Yes"], help="No = tidak ada edema; Some = edema ringan/resolved; Yes = edema berat/terus ada")

st.markdown("---")
st.subheader("🧪 Data Laboratorium")
col3, col4 = st.columns(2)

with col3:
    Bilirubin = st.number_input("Bilirubin (mg/dl)", min_value=0.0, step=0.1)
    Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, step=1)
    Albumin = st.number_input("Albumin (gm/dl)", min_value=0.0, step=0.1)
    Copper = st.number_input("Urine Copper (ug/day)", min_value=0, step=1)

with col4:
    Alk_Phos = st.number_input("Alkaline Phosphatase (U/liter)", min_value=0.0, step=1.0)
    SGOT = st.number_input("SGOT (U/ml)", min_value=0.0, step=1.0)
    Tryglicerides = st.number_input("Tryglicerides", min_value=0, step=1)
    Platelets = st.number_input("Platelets (x1000 per cubic ml)", min_value=0, step=1)
    Prothrombin = st.number_input("Prothrombin time (s)", min_value=0.0, step=0.1)

st.markdown("---")

# -----------------------------
# Prediksi
# -----------------------------
if st.button("🔍 Prediksi Status Pasien"):

    # Cek semua field sudah diisi
    if (N_Days == 0 or Age == 0 or
        Sex == "-- Pilih --" or Drug == "-- Pilih --" or
        Ascites == "-- Pilih --" or Hepatomegaly == "-- Pilih --" or
        Spiders == "-- Pilih --" or Edema == "-- Pilih --"):
        st.warning("Silakan lengkapi semua field sebelum prediksi!")
    else:
        kategori = encode(Drug, Sex, Ascites, Hepatomegaly, Spiders, Edema)

        input_data = [
            N_Days,
            kategori[0],  # Drug
            Age,
            kategori[1],  # Sex
            kategori[2],  # Ascites
            kategori[3],  # Hepatomegaly
            kategori[4],  # Spiders
            kategori[5],  # Edema
            Bilirubin,
            Cholesterol,
            Albumin,
            Copper,
            Alk_Phos,
            SGOT,
            Tryglicerides,
            Platelets,
            Prothrombin
        ]

        pred = predict_bagging(np.array(input_data))

        label_map = {
            0: "🟢 C – Pasien diprediksi masih bertahan hidup tanpa perlu transplantasi hati",
            1: "🟡 CL – Pasien diprediksi tetap hidup, tetapi berpotensi memerlukan transplantasi hati",
            2: "🔴 D – Pasien diprediksi memiliki risiko kematian lebih tinggi"
        }

        st.subheader("📌 Hasil Prediksi")
        st.success(label_map[pred])
