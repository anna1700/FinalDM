import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Framingham Heart Prediction",
    page_icon="❤️",
    layout="centered"
)

# ==============================================================================
# 2. LOAD DATA & TRAIN MODEL (PROSES DI BELAKANG LAYAR)
# ==============================================================================
@st.cache_data # Cache agar tidak training ulang setiap kali klik
def load_and_train():
    try:
        # Load Data
        df = pd.read_csv('framingham.csv')
        # PENTING: Framingham banyak data kosong (NaN), harus didrop
        df = df.dropna()
    except FileNotFoundError:
        return None, None, None

    # --- FITUR YANG DIPAKAI ---
    # Kita pilih 6 Fitur Utama biar Input User tidak terlalu ribet tapi akurat
    # 1. age (Umur)
    # 2. male (Gender: 1=Pria, 0=Wanita)
    # 3. currentSmoker (Perokok: 1=Ya, 0=Tidak)
    # 4. sysBP (Tekanan Darah Sistolik)
    # 5. glucose (Gula Darah)
    # 6. BMI (Berat/Tinggi)
    
    feature_cols = ['age', 'male', 'currentSmoker', 'sysBP', 'glucose', 'BMI']
    
    X = df[feature_cols]
    y_class = df['TenYearCHD'] # Target Klasifikasi
    y_reg = df['totChol']      # Target Regresi

    # --- MODEL 1: KLASIFIKASI (DECISION TREE) ---
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y_class)

    # --- MODEL 2: REGRESI (LINEAR REGRESSION) ---
    reg = LinearRegression()
    reg.fit(X, y_reg)

    return clf, reg, df

# Eksekusi Fungsi
model_clf, model_reg, df_clean = load_and_train()

# Cek Error File
if model_clf is None:
    st.error("❌ File 'framingham.csv' tidak ditemukan di Repository!")
    st.info("Pastikan file csv diupload ke GitHub di folder yang sama dengan app.py")
    st.stop()

# ==============================================================================
# 3. TAMPILAN SIDEBAR (INPUT DATA)
# ==============================================================================
st.sidebar.header("📝 Data Pasien")
st.sidebar.write("Masukkan parameter kesehatan:")

# Input Widgets
input_age = st.sidebar.slider("Umur (Tahun)", 20, 80, 50)
input_sex = st.sidebar.selectbox("Jenis Kelamin", ["Wanita", "Pria"])
input_smoker = st.sidebar.radio("Apakah Merokok?", ["Tidak", "Ya"])
input_sysBP = st.sidebar.number_input("Tekanan Darah (sysBP)", 90, 220, 120)
input_glucose = st.sidebar.number_input("Gula Darah (mg/dL)", 40, 400, 80)
input_bmi = st.sidebar.number_input("BMI (Body Mass Index)", 15.0, 50.0, 25.0)

# Konversi Input ke Angka (Sesuai Format Dataset Framingham)
# Dataset: male (1=Pria, 0=Wanita)
sex_val = 1 if input_sex == "Pria" else 0
# Dataset: currentSmoker (1=Ya, 0=Tidak)
smoker_val = 1 if input_smoker == "Ya" else 0

# Gabungkan jadi Array (Urutan HARUS SAMA dengan feature_cols di atas)
input_data = np.array([[input_age, sex_val, smoker_val, input_sysBP, input_glucose, input_bmi]])

# ==============================================================================
# 4. HALAMAN UTAMA
# ==============================================================================
st.title("❤️ Prediksi Jantung Framingham")
st.markdown(f"""
Aplikasi ini menggunakan Machine Learning dengan dataset **Framingham Heart Study** (Total Data Bersih: **{len(df_clean)}** pasien).
""")

# Tabs
tab1, tab2 = st.tabs(["🔍 Cek Risiko (Klasifikasi)", "📊 Cek Kolesterol (Regresi)"])

# --- TAB 1: PREDIKSI RISIKO JANTUNG ---
with tab1:
    st.subheader("Prediksi Risiko 10 Tahun")
    st.write("Apakah pasien berisiko terkena penyakit jantung koroner dalam 10 tahun ke depan?")
    
    if st.button("Analisis Risiko", type="primary"):
        # Prediksi Klasifikasi
        prediksi = model_clf.predict(input_data)[0]
        probabilitas = model_clf.predict_proba(input_data)[0][1] # Ambil % kemungkinan sakit
        
        st.divider()
        if prediksi == 1:
            st.error(f"⚠️ **HASIL: BERISIKO TINGGI**")
            st.write(f"Probabilitas: **{probabilitas*100:.1f}%**")
            st.warning("Saran: Segera konsultasikan ke dokter kardiologi.")
        else:
            st.success(f"✅ **HASIL: AMAN / RISIKO RENDAH**")
            st.write(f"Probabilitas: **{probabilitas*100:.1f}%**")
            st.info("Saran: Pertahankan pola hidup sehat.")

# --- TAB 2: PREDIKSI KOLESTEROL ---
with tab2:
    st.subheader("Estimasi Total Kolesterol")
    st.write("Memperkirakan kadar kolesterol total tanpa tes laboratorium.")
    
    if st.button("Hitung Estimasi"):
        # Prediksi Regresi
        est_chol = model_reg.predict(input_data)[0]
        
        st.divider()
        st.metric(label="Estimasi Kolesterol (mg/dL)", value=f"{est_chol:.2f}")
        
        # Indikator Warna
        if est_chol < 200:
            st.success("Kategori: NORMAL (< 200)")
        elif est_chol < 240:
            st.warning("Kategori: BATAS TINGGI (200-239)")
        else:
            st.error("Kategori: TINGGI (≥ 240)")
            st.write("⚠️ Angka ini cukup tinggi. Disarankan cek lab untuk kepastian.")

# Footer
st.markdown("---")
st.caption("Tugas Besar Data Mining - Metode Decision Tree & Linear Regression")
