import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# ==============================================================================
# 1. LOAD DATASET (Pastikan file framingham.csv ada di folder yang sama!)
# ==============================================================================
try:
    df = pd.read_csv('framingham.csv')
    df = df.dropna() # Hapus data kosong biar aman
except FileNotFoundError:
    st.error("❌ ERROR FATAL: File 'framingham.csv' tidak ditemukan! Taruh file csv di folder yang sama dengan app.py")
    st.stop()

# ==============================================================================
# 2. TRAINING MODEL OTOMATIS
# ==============================================================================
# --- Model 1: Klasifikasi (Risiko Jantung) ---
# Fitur: Age, sysBP, glucose, BMI (Kita sederhanakan jadi 4 fitur inti biar gak error input)
X_clf = df[['age', 'sysBP', 'glucose', 'BMI']]
y_clf = df['TenYearCHD']
model_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
model_clf.fit(X_clf, y_clf)

# --- Model 2: Regresi (Prediksi Kolesterol) ---
# Fitur: Age, sysBP, glucose, BMI
X_reg = df[['age', 'sysBP', 'glucose', 'BMI']]
y_reg = df['totChol']
model_reg = LinearRegression()
model_reg.fit(X_reg, y_reg)

# ==============================================================================
# 3. TAMPILAN WEB STREAMLIT
# ==============================================================================
st.title("Aplikasi Jantung Framingham ❤️")
st.success(f"Status: Terhubung ke Dataset ({len(df)} Data)")

# INPUT USER
st.sidebar.header("Masukkan Data Pasien")
age = st.sidebar.number_input("Umur (Tahun)", 20, 100, 50)
sysBP = st.sidebar.number_input("Tekanan Darah (sysBP)", 90, 250, 120)
glucose = st.sidebar.number_input("Gula Darah (mg/dL)", 50, 400, 80)
bmi = st.sidebar.number_input("BMI (Berat/Tinggi)", 15.0, 50.0, 25.0)

# SUSUN DATA INPUT (Urutannya harus SAMA PERSIS dengan X_clf di atas)
input_data = np.array([[age, sysBP, glucose, bmi]])

# TAB HASIL
tab1, tab2 = st.tabs(["🔍 Cek Risiko (Klasifikasi)", "📈 Cek Kolesterol (Regresi)"])

with tab1:
    st.subheader("Prediksi Risiko Jantung 10 Tahun")
    if st.button("Analisis Risiko"):
        prediksi = model_clf.predict(input_data)[0]
        probabilitas = model_clf.predict_proba(input_data)[0][1]
        
        if prediksi == 1:
            st.error(f"⚠️ BERISIKO TINGGI (Probabilitas: {probabilitas*100:.1f}%)")
            st.write("Saran: Segera konsultasi ke dokter.")
        else:
            st.success(f"✅ AMAN (Probabilitas: {probabilitas*100:.1f}%)")
            st.write("Saran: Jaga pola hidup.")

with tab2:
    st.subheader("Estimasi Kolesterol")
    if st.button("Hitung Kolesterol"):
        hasil_chol = model_reg.predict(input_data)[0]
        st.info(f"Estimasi Total Kolesterol: **{hasil_chol:.2f} mg/dL**")
