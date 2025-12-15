import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier # <-- Ganti Import ini
from sklearn.linear_model import LinearRegression

# Config
st.set_page_config(page_title="Heart Prediction RF", page_icon="🌲")

# 1. LOAD & TRAIN
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv('framingham.csv').dropna()
    except FileNotFoundError:
        return None, None, None

    # Fitur Utama
    feature_cols = ['age', 'male', 'currentSmoker', 'sysBP', 'glucose', 'BMI']
    X = df[feature_cols]
    y_class = df['TenYearCHD']
    y_reg = df['totChol']

    # --- MODEL 1: RANDOM FOREST ---
    # Kita pakai 100 pohon (n_estimators=100)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_class)

    # --- MODEL 2: LINEAR REGRESSION ---
    reg = LinearRegression()
    reg.fit(X, y_reg)

    return clf, reg, df

model_clf, model_reg, df = load_and_train()

if model_clf is None:
    st.error("File 'framingham.csv' tidak ditemukan!")
    st.stop()

# 2. SIDEBAR
st.sidebar.header("Data Pasien")
age = st.sidebar.slider("Umur", 20, 90, 50)
sex = st.sidebar.selectbox("Gender", ["Wanita", "Pria"])
smoke = st.sidebar.radio("Perokok?", ["Tidak", "Ya"])
sysBP = st.sidebar.number_input("Tensi (sysBP)", 90, 220, 120)
glucose = st.sidebar.number_input("Gula Darah", 50, 400, 80)
bmi = st.sidebar.number_input("BMI", 15.0, 50.0, 25.0)

# Konversi
val_sex = 1 if sex == "Pria" else 0
val_smoke = 1 if smoke == "Ya" else 0
input_data = np.array([[age, val_sex, val_smoke, sysBP, glucose, bmi]])

# 3. UTAMA
st.title("Aplikasi Jantung (Random Forest)")
st.write("Menggunakan algoritma **Random Forest** yang lebih akurat.")

tab1, tab2 = st.tabs(["🌲 Cek Risiko (RF)", "📈 Cek Kolesterol (Regresi)"])

with tab1:
    st.subheader("Prediksi Risiko Jantung")
    if st.button("Analisis Risiko"):
        pred = model_clf.predict(input_data)[0]
        prob = model_clf.predict_proba(input_data)[0][1]
        
        if pred == 1:
            st.error(f"⚠️ BERISIKO TINGGI ({prob*100:.1f}%)")
        else:
            st.success(f"✅ AMAN / RENDAH ({prob*100:.1f}%)")

with tab2:
    st.subheader("Estimasi Kolesterol")
    if st.button("Hitung"):
        chol = model_reg.predict(input_data)[0]
        st.info(f"Estimasi: {chol:.2f} mg/dL")
