import streamlit as st
import numpy as np
import librosa
import joblib
import io
from audiorecorder import audiorecorder
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="DTW Audio Recognizer", page_icon="📏")

# --- Load Data Referensi ---
@st.cache_resource
def load_references():
    try:
        return joblib.load("dtw_references.pkl")
    except FileNotFoundError:
        st.error("File 'dtw_references.pkl' tidak ditemukan. Jalankan process_references.py dulu.")
        return None

references = load_references()

# --- Fungsi Ekstraksi Fitur (Sama persis dengan process_references.py) ---
def extract_features_dtw(y, sr=22050):
    try:
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y_trimmed) < 1024:
            return None
            
        # Ekstrak MFCC tanpa rata-rata
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        return mfcc.T # Transpose agar (Waktu, Fitur)
    except Exception:
        return None

# --- Logika Perhitungan Jarak DTW ---
def predict_with_dtw(input_features, reference_data):
    min_distance = float('inf')
    prediction = "noise"
    
    # Bandingkan dengan semua referensi 'open'
    for ref in reference_data['open']:
        # Hitung jarak DTW
        distance, _ = fastdtw(input_features, ref, dist=euclidean)
        if distance < min_distance:
            min_distance = distance
            prediction = "open"
            
    # Bandingkan dengan semua referensi 'close'
    for ref in reference_data['close']:
        distance, _ = fastdtw(input_features, ref, dist=euclidean)
        if distance < min_distance:
            min_distance = distance
            prediction = "close"
            
    return prediction, min_distance

# --- UI Streamlit ---
st.title("📏 Identifikasi Suara dengan DTW")
st.write("Membandingkan suara Anda dengan 200 data referensi menggunakan Dynamic Time Warping.")

# CSS untuk status
st.markdown("""
<style>
.status-box { padding: 20px; border-radius: 10px; text-align: center; color: white; font-weight: bold; font-size: 20px; }
.open { background-color: #28a745; }
.close { background-color: #dc3545; }
.noise { background-color: #6c757d; }
</style>
""", unsafe_allow_html=True)

if references is not None:
    audio = audiorecorder("Rekam Suara", "Stop")

    if len(audio) > 0:
        st.write("Menganalisis jarak DTW...")
        
        # 1. Proses Audio Input
        audio_bytes = audio.export().read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # 2. Ekstrak Fitur Time Series
        input_features = extract_features_dtw(y, sr)
        
        if input_features is None:
            st.warning("Suara tidak terdeteksi atau terlalu pendek.")
        else:
            # 3. Hitung Jarak dengan 200 Data
            pred_label, distance = predict_with_dtw(input_features, references)
            
            # 4. Tampilkan Hasil
            st.write(f"Jarak terdekat yang ditemukan: **{distance:.2f}**")
            
            if pred_label == "open":
                st.markdown('<div class="status-box open">Opening the door now</div>', unsafe_allow_html=True)
            elif pred_label == "close":
                st.markdown('<div class="status-box close">The door will be closed soon</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box noise">Tidak Dikenali</div>', unsafe_allow_html=True)