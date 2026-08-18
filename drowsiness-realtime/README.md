# Real-Time Drowsiness Detection

Aplikasi deteksi kantuk pengemudi secara real-time via webcam menggunakan model **LandmarkGRU** (EAR + MAR) dan **MediaPipe Tasks Face Landmarker**.

Aplikasi ini menggunakan checkpoint model yang **sudah dilatih** (`best_landmark_gru_ear_mar.pth`), bukan melatih model baru.

---

## 1. Spesifikasi Model

- **Arsitektur**: `LandmarkGRU` (2-Layer GRU + Temporal Attention)
- **Input Dimensi**: 2 (`[EAR, MAR]`)
- **Hidden Dimension**: 128
- **Sequence Length**: 60 frame (~2 detik pada 30 FPS)
- **Ekstraksi Fitur**: EAR V3 + MAR dari 478 MediaPipe Facial Landmarks
- **Normalisasi**: Z-Score menggunakan *Training Mean & Std* dari checkpoint
- **Output Kelas**:
  - `0`: **NOT DROWSY**
  - `1`: **DROWSY**
- **Checkpoint**: `models/best_landmark_gru_ear_mar.pth`
- **Konfigurasi**: `models/FINAL_landmark_gru_config.json`
- **MediaPipe Task**: `mediapipe/face_landmarker.task`

---

## 2. Struktur Direktori

```text
drowsiness-realtime/
│
├── models/
│   ├── best_landmark_gru_ear_mar.pth
│   └── FINAL_landmark_gru_config.json
│
├── mediapipe/
│   └── face_landmarker.task
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── landmark_detector.py
│   ├── feature_extractor.py
│   ├── preprocessing.py
│   ├── inference.py
│   ├── smoothing.py
│   └── realtime.py
│
├── tests/
│   └── test_model.py
│
├── logs/
│   └── realtime_predictions.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Instalasi

### Prasyarat
- Python 3.10, 3.11, atau 3.12
- Webcam yang terhubung ke komputer

### Langkah Instalasi (Windows PowerShell)

```powershell
# 1. Buat virtual environment
python -m venv .venv

# 2. Aktifkan virtual environment
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 4. Pengujian Model (Sanity Test)

Sebelum menjalankan kamera, pastikan model dan checkpoint dapat dimuat tanpa error dengan menjalankan unit test:

```powershell
python tests/test_model.py
```

---

## 5. Menjalankan Aplikasi Real-Time

### Perintah Standar

```powershell
python -m src.realtime
```

### Opsi Argumen CLI

| Argumen | Default | Deskripsi |
| :--- | :--- | :--- |
| `--camera-index` | `0` | Indeks webcam OpenCV (coba `1` atau `2` jika menggunakan kamera eksternal) |
| `--threshold` | `0.50` | Ambang batas probabilitas untuk klasifikasi `DROWSY` |
| `--smoothing-window` | `5` | Jumlah frame untuk moving average smoothing |
| `--drowsy-alert-sec` | `4.0` | Batas waktu kantuk umum (detik) sebelum suara alarm aktif |
| `--nodding-alert-sec` | `1.5` | Batas waktu kepala nodding / menunduk terkulai (detik) sebelum alarm aktif |
| `--look-aside-alert-sec` | `3.0` | Batas waktu menengok kiri/kanan (detik) sebelum alarm distraksi aktif |
| `--microsleep-alert-sec` | `1.0` | Batas waktu mata terpejam / microsleep (detik) sebelum alarm aktif cepat |
| `--yawn-alert-sec` | `4.5` | Batas waktu menguap (detik) sebelum alarm suara aktif (toleransi menguap) |
| `--eye-thresh` | `0.21` | Batas EAR mata terpejam |
| `--yawn-thresh` | `0.45` | Batas MAR menguap |
| `--nodding-thresh` | `0.45` | Batas rasio vertikal wajah untuk deteksi kepala nodding (dikalibrasi) |
| `--look-aside-thresh` | `0.28` | Batas rasio rotasi horizontal untuk menengok kiri/kanan |
| `--no-sound` | `False` | Nonaktifkan bunyi alarm audio |
| `--device` | `cpu` | Device inferensi (`cpu` atau `cuda`) |
| `--log` | `False` | Simpan riwayat deteksi ke file CSV di `logs/realtime_predictions.csv` |
| `--no-display` | `False` | Mode headless (tanpa GUI OpenCV) |

#### Contoh Perintah Kustom:
```powershell
# Menjalankan dengan konfigurasi bawaan (Kantuk: 4s, Nodding: 1.5s, Menengok: 3s)
python -m src.realtime

# Menyesuaikan parameter secara manual
python -m src.realtime --drowsy-alert-sec 4.0 --nodding-alert-sec 1.5 --look-aside-alert-sec 3.0
```

---

## 6. Skenario Kantuk Realistis & Alarm

Sistem dilengkapi skenario kontekstual berbasis fisiologis pengemudi:

1. **Skenario 1 — Head Nodding / Kepala Menunduk Terkulai (`NODDING`)**:
   - **Kondisi**: Rasio posisi wajah dahi-hidung-dagu < 0.45 (atau < 0.52 disertai mata sayu). Telah dikalibrasi agar **tidak agresif** terhadap posisi duduk/mengetik biasa.
   - **Perilaku**: Alarm suara berbunyi setelah **1.5 detik** berturut-turut.
   - **Indikator**: Status `NODDING (DROWSY)` berwarna Merah.

2. **Skenario 2 — Menengok Kiri / Kanan (`LOOKING ASIDE`)**:
   - **Kondisi**: Rotasi horizontal kepala (Yaw Ratio >= 0.38) menjauh dari arah jalan.
   - **Perilaku**: Deteksi distraksi pengemudi. Alarm berbunyi jika menengok ke samping selama **3.0 detik** berturut-turut.
   - **Indikator**: Status `LOOKING LEFT/RIGHT (DISTRACTED)` berwarna Oranye-Merah.

3. **Skenario 3 — Pola Kantuk Sekuensial GRU (`DROWSY`)**:
   - **Kondisi**: Model LandmarkGRU mendeteksi kombinasi temporal kantuk (kedipan lambat, mata sayu, dsb).
   - **Perilaku**: Alarm suara berbunyi setelah kondisi kantuk bertahan **4.0 detik** berturut-turut.
   - **Indikator**: Status `DROWSY` berwarna Merah.

4. **Skenario 4 — Microsleep / Mata Terpejam Kritis (`MICROSLEEP`)**:
   - **Kondisi**: EAR < 0.21 (mata tertutup total / sayu berat).
   - **Perilaku**: Respon alarm tercepat (berbunyi jika mata terpejam >= **1.0 detik**).
   - **Indikator**: Status `MICROSLEEP (CRITICAL)` berwarna Merah Pekat.

5. **Skenario 5 — Menguap / Peringatan Dini (`YAWNING`)**:
   - **Kondisi**: MAR >= 0.45 (mulut terbuka lebar) dan mata masih terbuka.
   - **Perilaku**: Menampilkan status peringatan Oranye tanpa langsung membunyikan alarm panik. Alarm baru berbunyi jika menguap berlangsung tidak wajar (> **4.5 detik**).
   - **Indikator**: Status `YAWNING (WARNING)` berwarna Oranye.

6. **Skenario 6 — Pengemudi Sadar (`NOT DROWSY`)**:
   - **Kondisi**: Posisi kepala tegak ke depan, mata dan mulut dalam rentang normal sadar.
   - **Perilaku**: Status Hijau, timer alarm di-reset otomatis ke 0.0s.

Untuk keluar dari aplikasi, tekan tombol **`q`** pada keyboard.

---

## 7. Catatan Evaluasi & Integritas Ilmiah

- Model ini merupakan **baseline terbaik yang teramati dalam eksperimen offline internal** (LOSO Cross-Validation F1 ~ 0.6143).
- **UNKNOWN State**: Jika wajah tidak terdeteksi atau nilai EAR/MAR tidak valid (> 1.0), sistem sengaja menandai status sebagai `UNKNOWN` dan tidak memaksakan prediksi `DROWSY` atau `NOT DROWSY`.
- Tidak ada klaim performa/akurasi real-time absolut sebelum dievaluasi menggunakan dataset benchmark real-time beranotasi *ground truth*.
