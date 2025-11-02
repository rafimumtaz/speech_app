import streamlit as st
import numpy as np
import librosa
import joblib
import io
from audiorecorder import audiorecorder # (BARU) Impor library baru

# --- Konfigurasi Halaman & CSS (Sama seperti sebelumnya) ---
st.set_page_config(
    page_title="Detektor Perintah Pintu",
    page_icon="🎙️",
    layout="centered"
)
st.markdown(
    """
    <style>
    /* Kontainer Status */
    .status-container {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        transition: all 0.3s ease-in-out;
    }
    .status-open {
        background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;
    }
    .status-close {
        background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;
    }
    .status-noise {
        background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db;
    }
    .status-text { font-size: 1.5em; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model dan Scaler ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("speech_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model 'speech_model.pkl' atau 'scaler.pkl' tidak ditemukan.")
        st.error("Harap jalankan 'train_model.py' terlebih dahulu.")
        return None, None

model, scaler = load_models()

RMS_THRESHOLD = 0.005 # Ambang batas keheningan

# --- Fungsi Ekstraksi Fitur (Sama seperti v10) ---
def extract_features(y, sr=22050):
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        rms_val = np.mean(librosa.feature.rms(y=y_trimmed))
        
        if rms_val < RMS_THRESHOLD:
            return 'below_threshold'
        
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y_trimmed))
        features = np.hstack((mfccs_mean, zcr_mean, rms_val))
        return features
    except Exception:
        return None

# --- Tampilan UI (DIPERBARUI) ---
st.title("🎙️ Detektor Perintah Pintu")
st.write("Klik ikon mikrofon untuk merekam perintah 'Open' atau 'Close', lalu klik lagi untuk berhenti.")

# Label kelas (sesuai train_model.py)
class_labels = {0: "close", 1: "noise", 2: "open"}

if model is not None:
    # (BARU) Komponen Perekam Audio
    audio = audiorecorder("Klik untuk Berbicara", "Merekam... (Klik lagi untuk stop)")

    # Tempat untuk menampilkan status
    status_placeholder = st.empty()

    if len(audio) > 0:
        # Jika ada audio baru yang direkam
        st.write("Menganalisis audio...")
        
        # 1. Konversi audio bytes dari perekam ke numpy array
        # Kita gunakan io.BytesIO untuk membacanya di memori
        audio_bytes = audio.export().read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # 2. Ekstrak Fitur
        features = extract_features(y, sr)
        
        # 3. Prediksi
        if features is None:
            prediction_label = "noise"
        elif isinstance(features, str) and features == 'below_threshold':
            prediction_label = "noise"
        else:
            features_scaled = scaler.transform([features])
            prediction_idx = model.predict(features_scaled)[0]
            prediction_label = class_labels[prediction_idx]

        # 4. Tampilkan hasil
        if prediction_label == "open":
            status_placeholder.markdown(
                '<div class="status-container status-open"><span class="status-text">Opening the door now</span></div>',
                unsafe_allow_html=True
            )
        elif prediction_label == "close":
            status_placeholder.markdown(
                '<div class="status-container status-close"><span class="status-text">The door will be closed soon</span></div>',
                unsafe_allow_html=True
            )
        else: # "noise"
            status_placeholder.markdown(
                '<div class="status-container status-noise"><span class="status-text">Perintah tidak dikenali (Noise).</span></div>',
                unsafe_allow_html=True
            )
        
        # Tampilkan pemutar audio untuk debugging (opsional)
        st.audio(audio_bytes)
else:
    # Tampilan default saat menunggu rekaman
    status_placeholder.markdown(
        '<div class="status-container status-noise"><span class="status-text">...</span></div>',
        unsafe_allow_html=True
    )