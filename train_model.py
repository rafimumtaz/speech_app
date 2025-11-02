import librosa
import numpy as np
import pandas as pd
import os
import glob
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- (BARU) Fungsi Augmentasi ---
def augment_data(y, sr, noise_samples):
    augmented_data = []
    
    # 1. Tambah Noise (dari folder noise Anda)
    if noise_samples:
        noise = random.choice(noise_samples)
        noise_level = np.random.uniform(0.001, 0.005)
        # Samakan panjang audio
        if len(y) > len(noise):
            y = y[:len(noise)]
        else:
            noise = noise[:len(y)]
        y_noise = y + (noise * noise_level)
        augmented_data.append(y_noise)
    
    # 2. Geser Waktu (Time Shift)
    shift_range = int(np.random.uniform(-0.1, 0.1) * sr)
    y_shift = np.roll(y, shift_range)
    augmented_data.append(y_shift)
    
    # 3. Ubah Pitch (sedikit)
    pitch_factor = np.random.uniform(-0.5, 0.5) # Perubahan pitch kecil
    y_pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_factor)
    augmented_data.append(y_pitch)
    
    return augmented_data

# --- (DIPERBARUI) Fungsi Ekstraksi Fitur dengan Trimming ---
def extract_features(y, sr=22050):
    try:
        # (BARU) Potong keheningan dari awal & akhir
        # Ini memastikan kita hanya menganalisis suaranya
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        
        # Jika klipnya terlalu pendek setelah dipotong, lewati
        if len(y_trimmed) < 2048: # (sekitar 0.1 detik)
            return None

        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y_trimmed))
        
        rms_mean = np.mean(librosa.feature.rms(y=y_trimmed))
        
        features = np.hstack((mfccs_mean, zcr_mean, rms_mean))
        return features
    except Exception as e:
        print(f"Error mengekstrak fitur: {e}")
        return None

# --- (DIPERBARUI) Fungsi Load Data dengan Augmentasi ---
def load_data(max_samples_per_class=1000):
    features = []
    labels = []
    
    # Label: 0 = close, 1 = noise, 2 = open
    
    # 1. Load Noise dulu (untuk digunakan di augmentasi)
    print("Memuat sampel noise...")
    noise_samples_raw = []
    for file_path in glob.glob("noise/*.wav"):
        try:
            y, sr = librosa.load(file_path)
            noise_samples_raw.append(y)
        except Exception:
            continue
    
    # 2. Load data 'noise' (batasi jumlahnya)
    print("Memproses data 'noise'...")
    random.shuffle(noise_samples_raw) # Acak noise
    count = 0
    for y_noise in noise_samples_raw:
        if count >= max_samples_per_class:
            break
        feat = extract_features(y_noise)
        if feat is not None:
            features.append(feat)
            labels.append(1) # 1 untuk noise
            count += 1
    
    print(f"Data noise yang digunakan: {count}")

    # 3. Load dan Augmentasi data 'close'
    print("Memproses dan meng-augmentasi data 'close'...")
    close_files = glob.glob("close/*.wav")
    for file_path in close_files:
        try:
            y, sr = librosa.load(file_path, duration=2.0)
            
            # Tambahkan data asli
            feat_orig = extract_features(y, sr)
            if feat_orig is not None:
                features.append(feat_orig)
                labels.append(0) # 0 untuk close
            
            # Tambahkan data augmentasi (buat 10 variasi per file)
            for _ in range(10): 
                y_aug_list = augment_data(y, sr, noise_samples_raw)
                for y_aug in y_aug_list:
                    feat_aug = extract_features(y_aug, sr)
                    if feat_aug is not None:
                        features.append(feat_aug)
                        labels.append(0)
                        
        except Exception as e:
            print(f"Gagal memproses {file_path}: {e}")
            
    # 4. Load dan Augmentasi data 'open'
    print("Memproses dan meng-augmentasi data 'open'...")
    open_files = glob.glob("open/*.wav")
    for file_path in open_files:
        try:
            y, sr = librosa.load(file_path, duration=2.0)
            
            # Tambahkan data asli
            feat_orig = extract_features(y, sr)
            if feat_orig is not None:
                features.append(feat_orig)
                labels.append(2) # 2 untuk open
            
            # Tambahkan data augmentasi (buat 10 variasi per file)
            for _ in range(10): 
                y_aug_list = augment_data(y, sr, noise_samples_raw)
                for y_aug in y_aug_list:
                    feat_aug = extract_features(y_aug, sr)
                    if feat_aug is not None:
                        features.append(feat_aug)
                        labels.append(2)

        except Exception as e:
            print(f"Gagal memproses {file_path}: {e}")

    print(f"Total data setelah augmentasi: {len(features)}")
    return pd.DataFrame(features), np.array(labels)

# --- Main Script ---
if __name__ == "__main__":
    
    # 1. Load dan ekstrak fitur
    print("Memulai ekstraksi fitur dan augmentasi...")
    # Batasi data noise agar seimbang dengan data augmentasi
    X, y = load_data(max_samples_per_class=1000) 
    
    if X.empty or len(np.unique(y)) < 3:
        print("Data tidak cukup untuk melatih. Pastikan ada file di folder open, close, dan noise.")
    else:
        # 2. Pisahkan data
        print("Membagi data training dan testing...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 3. Scaling Data
        print("Melakukan scaling data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Latih Model
        print("Melatih model RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # 5. Evaluasi Model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n--- HASIL EVALUASI ---")
        print(f"Akurasi Model: {accuracy * 100:.2f}%")
        print("\nLaporan Klasifikasi:")
        print(classification_report(y_test, y_pred, target_names=['close (0)', 'noise (1)', 'open (2)']))
        
        # 6. Simpan Model dan Scaler
        print("Menyimpan model ke 'speech_model.pkl'...")
        joblib.dump(model, "speech_model.pkl")
        
        print("Menyimpan scaler ke 'scaler.pkl'...")
        joblib.dump(scaler, "scaler.pkl")
        
        print("\nPelatihan selesai!")