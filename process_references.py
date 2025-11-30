import librosa
import numpy as np
import os
import glob
import joblib

def extract_features_dtw(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, duration=2.0)
        # Potong hening (Trim)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) < 1024:
            return None

        # Ekstrak MFCC
        # Kita TIDAK melakukan np.mean di sini karena butuh data urutan waktu
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # TRANSPOSE PENTING: 
        # Librosa menghasilkan (n_mfcc, time). DTW butuh (time, n_mfcc).
        return mfcc.T 
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_references():
    references = {
        "open": [],
        "close": []
    }
    
    # Proses Open (Ambil 100 data)
    print("Memproses data referensi 'open'...")
    files = glob.glob("reference_data/open/*.wav")[:100]
    for f in files:
        feat = extract_features_dtw(f)
        if feat is not None:
            references["open"].append(feat)
            
    # Proses Close (Ambil 100 data)
    print("Memproses data referensi 'close'...")
    files = glob.glob("reference_data/close/*.wav")[:100]
    for f in files:
        feat = extract_features_dtw(f)
        if feat is not None:
            references["close"].append(feat)

    print(f"Selesai. Disimpan {len(references['open'])} open dan {len(references['close'])} close.")
    joblib.dump(references, "dtw_references.pkl")

if __name__ == "__main__":
    save_references()