import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(page_title="Framingham Heart Study App", page_icon="❤️")

# ==============================================================================
# 1. LOAD DATA & LATIH MODEL (DI BELAKANG LAYAR)
# ==============================================================================
@st.cache_data
def train_models():
    # Load Data
    try:
        df = pd.read_csv('framingham.csv')
        df = df.dropna() # Hapus data kosong agar akurat
    except:
        return None, None
    
    # --- MODEL 1: KLASIFIKASI (Risiko Jantung 10 Tahun) ---
    # Fitur yang dipakai: Umur, Gender, Rokok, Tensi(sysBP), Gula Darah(glucose), BMI
    X_clf = df[['age', 'male', 'currentSmoker', 'sysBP', 'glucose', 'BMI']]
    y_clf = df['TenYearCHD']
    
    model_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_clf.fit(X_clf, y_clf)
    
    # --- MODEL 2: REGRESI (Prediksi Kolesterol) ---
    # Fitur yang dipakai: Umur, BMI, Tensi(sysBP), Gula Darah(glucose)
    X_reg = df[['age', 'BMI', 'sysBP', 'glucose']]
    y_reg = df['totChol']
    
    model_reg = LinearRegression()
    model_reg.fit(X_reg, y_reg)
    
    return model_clf, model_reg

# Eksekusi fungsi training
model_clf, model_reg = train_models()

# Cek jika file tidak ada
if model_clf is None:
    st.error("❌ File 'framingham.csv' tidak ditemukan! Harap upload ke folder yang sama.")
    st.stop()

# ==============================================================================
# 2. TAMPILAN SIDEBAR (INPUT USER)
# ==============================================================================
st.sidebar.header("📝 Masukkan Data Pasien")
st.sidebar.write("Sesuaikan dengan kondisi fisik pasien:")

# Input Data
input_age = st.sidebar.slider("Umur (Tahun)", 20, 90, 45)
input_sex = st.sidebar.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
input_smoke = st.sidebar.radio("Apakah Perokok?", ["Tidak", "Ya"])
input_sysBP = st.sidebar.number_input("Tekanan Darah Sistolik (sysBP)", 90, 250, 120)
input_glucose = st.sidebar.number_input("Kadar Gula Darah (mg/dL)", 50, 400, 80)
input_bmi = st.sidebar.number_input("Indeks Massa Tubuh (BMI)", 15.0, 50.0, 25.0)

# Konversi Input User ke Angka (Biar dimengerti Komputer)
sex_encoded = 1 if input_sex == "Pria" else 0
smoke_encoded = 1 if input_smoke == "Ya" else 0

# ==============================================================================
# 3. HALAMAN UTAMA (TABS)
# ==============================================================================
st.title("❤️ Aplikasi Prediksi Jantung (Framingham)")
st.write(f"Menggunakan dataset **Framingham Heart Study** dengan **4.000+ data pasien**.")

tab1, tab2 = st.tabs(["🩺 Cek Risiko Jantung", "📊 Cek Estimasi Kolesterol"])

# --- TAB 1: KLASIFIKASI ---
with tab1:
    st.header("Prediksi Risiko 10 Tahun")
    st.write("Memprediksi apakah pasien berisiko terkena penyakit jantung koroner dalam 10 tahun ke depan.")
    
    if st.button("Analisis Risiko Sekarang"):
        # Siapkan data input sesuai urutan training
        input_data = np.array([[input_age, sex_encoded, smoke_encoded, input_sysBP, input_glucose, input_bmi]])
        
        # Prediksi
        prediksi = model_clf.predict(input_data)[0]
        probabilitas = model_clf.predict_proba(input_data)[0][1] # Ambil % kemungkinan sakit
        
        if prediksi == 1:
            st.error(f"⚠️ **BERISIKO TINGGI!**")
            st.write(f"Probabilitas risiko: **{probabilitas*100:.1f}%**")
            st.write("Saran: Segera konsultasi ke dokter kardiologi dan perbaiki gaya hidup.")
        else:
            st.success(f"✅ **AMAN / RISIKO RENDAH**")
            st.write(f"Probabilitas risiko: **{probabilitas*100:.1f}%**")
            st.write("Saran: Pertahankan pola hidup sehat.")

# --- TAB 2: REGRESI ---
with tab2:
    st.header("Estimasi Angka Kolesterol")
    st.write("Memperkirakan kadar **Total Kolesterol** tanpa tes darah, berdasarkan profil fisik.")
    
    if st.button("Hitung Estimasi Kolesterol"):
        # Siapkan data input (hanya fitur yang dipakai regresi)
        input_reg = np.array([[input_age, input_bmi, input_sysBP, input_glucose]])
        
        # Prediksi
        est_chol = model_reg.predict(input_reg)[0]
        
        st.metric(label="Estimasi Total Kolesterol", value=f"{est_chol:.2f} mg/dL")
        
        # Logika Warna
        if est_chol < 200:
            st.success("Kategori: NORMAL (< 200 mg/dL)")
        elif 200 <= est_chol < 240:
            st.warning("Kategori: AMBANG BATAS TINGGI (200-239 mg/dL)")
        else:
            st.error("Kategori: TINGGI (>= 240 mg/dL)")
            
# Footer
st.markdown("---")
st.caption("Dikembangkan untuk Tugas Besar Data Mining - Menggunakan Algoritma Decision Tree & Linear Regression.")