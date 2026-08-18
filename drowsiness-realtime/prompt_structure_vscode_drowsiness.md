# Struktur VS Code — Real-Time Drowsiness Detection

Dokumen ini berisi struktur project dan prompt untuk agent lain agar dapat membuat aplikasi real-time dari model yang **sudah dilatih**, bukan membuat model baru.

## Model utama

Checkpoint utama:

```text
models/best_landmark_gru_ear_mar.pth
```

Konfigurasi:

```text
models/FINAL_landmark_gru_config.json
```

Model utama:

- Input: `EAR + MAR`
- Sequence length: `60 frame`
- GRU: `2 layer`
- Hidden dimension: `128`
- Temporal attention
- Output:
  - `0 = NOT DROWSY`
  - `1 = DROWSY`

Model ini adalah **baseline terbaik yang teramati sejauh ini**, bukan klaim model terbaik secara universal.

---

# 1. Struktur Folder

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

# 2. Prompt — `src/model.py`

```text
BUAT FILE: src/model.py

Tujuan:
Membuat definisi arsitektur LandmarkGRU yang SAMA PERSIS dengan model
yang sudah digunakan saat training.

Checkpoint:
models/best_landmark_gru_ear_mar.pth

Konfigurasi:
- input_dim = 2
- hidden_dim = 128
- num_layers = 2
- num_classes = 2
- input features = [EAR, MAR]

Model menggunakan:
- 2-layer GRU
- hidden size 128
- temporal attention
- classifier untuk 2 kelas

Kelas:
0 = NOT DROWSY
1 = DROWSY

ATURAN:
1. Jangan membuat arsitektur baru.
2. Jangan mengganti jumlah GRU layer.
3. Jangan mengganti hidden dimension.
4. Jangan menambahkan dropout, BatchNorm, LSTM, Transformer,
   atau layer lain yang tidak ada pada model training.
5. Forward harus kompatibel dengan checkpoint.
6. Pastikan state_dict dari:
   best_landmark_gru_ear_mar.pth
   dapat di-load tanpa mismatch.
7. Sediakan helper:
   - create_model()
   - load_model(checkpoint_path, device)

load_model harus:
- membaca konfigurasi;
- membuat model;
- load state_dict;
- model.eval();
- memberikan error jelas jika checkpoint tidak cocok.

Jangan melakukan training di file ini.
```

---

# 3. Prompt — `src/landmark_detector.py`

```text
BUAT FILE: src/landmark_detector.py

Tujuan:
Mendeteksi wajah menggunakan MediaPipe Face Landmarker secara realtime.

Gunakan:
MediaPipe Tasks / Face Landmarker API modern.

JANGAN menggunakan:
mp.solutions.face_mesh

Model MediaPipe:
mediapipe/face_landmarker.task

Input:
- frame OpenCV BGR

Output:
- face landmarks
- bounding information jika diperlukan
- valid / invalid status

Buat fungsi:

detect_face_landmarks(frame)

Fungsi harus:
1. menerima frame BGR;
2. mengubah ke RGB;
3. membuat MediaPipe Image;
4. menjalankan Face Landmarker;
5. mengambil landmarks;
6. mengembalikan landmarks jika valid.

Validasi:
- tidak ada wajah -> invalid
- landmark kosong -> invalid
- lebih dari satu wajah -> ikuti konfigurasi satu wajah atau tandai invalid

Tambahkan error handling.

Jangan menghitung EAR/MAR di file ini.

Tugas:
WEBCAM FRAME -> LANDMARKS
```

---

# 4. Prompt — `src/feature_extractor.py`

```text
BUAT FILE: src/feature_extractor.py

Tujuan:
Mengubah facial landmarks dari MediaPipe menjadi:
- EAR
- MAR

HARUS menggunakan implementasi EAR V3 yang SAMA dengan training.

Jangan menggunakan EAR V2.

Diagnostic EAR V3 sebelumnya:
- EAR sekitar 0.10–0.44
- EAR > 1 dianggap tidak valid
- MAR > 1 dianggap tidak valid

ATURAN:
1. Jangan mengubah landmark indices.
2. Jangan mengarang indeks baru.
3. Jika source code EAR V3 dari notebook training tersedia,
   gunakan persis.
4. Jika indeks/formula EAR V3 tidak tersedia,
   BERHENTI dan minta source EAR V3 asli.
5. Jangan menggunakan formula berbeda tanpa alasan.

Buat:

extract_features(landmarks)

Output:

{
    "ear": float,
    "mar": float,
    "valid": bool
}

Validasi:
- EAR finite
- MAR finite
- tidak negatif
- EAR > 1 -> invalid
- MAR > 1 -> invalid

Jangan melakukan normalization di file ini.

Tugas:
LANDMARKS -> RAW EAR + MAR
```

---

# 5. Prompt — `src/preprocessing.py`

```text
BUAT FILE: src/preprocessing.py

Tujuan:
Melakukan preprocessing EAR + MAR yang SAMA dengan training.

Model:
best_landmark_gru_ear_mar.pth

Input per frame:
[EAR, MAR]

Sequence length:
60 frame

Normalisasi:
z-score menggunakan TRAINING mean dan std.

Rumus:

normalized = (feature - train_mean) / train_std

JANGAN:
- menghitung mean dari webcam;
- menghitung std dari webcam;
- menghitung statistik dari sequence realtime;
- menggunakan Relative-EAR;
- menggunakan PERCLOS;
- menggunakan delta;
- menggunakan normalization baru.

Statistik harus berasal dari training/config.

Cari:
FINAL_landmark_gru_config.json

Jika mean/std tersedia:
gunakan nilainya.

Jika mean/std tidak tersedia:
BERHENTI.
Jangan menebak.

Buat class:

SequenceBuffer

Methods:
- append([ear, mar])
- len()
- is_ready()
- get_sequence()
- clear()

Sequence:
(60, 2)

Sebelum PyTorch:
(60, 2) -> (1, 60, 2)

Output:
torch.FloatTensor

Validasi:
- tidak boleh NaN
- tidak boleh inf
- shape harus tepat
```

---

# 6. Prompt — `src/inference.py`

```text
BUAT FILE: src/inference.py

Tujuan:
Menjalankan inference model yang sudah dilatih.

Checkpoint:
models/best_landmark_gru_ear_mar.pth

Model:
Raw EAR + MAR LandmarkGRU

Flow:

sequence (1,60,2)
    ->
model
    ->
logits
    ->
softmax
    ->
probability

Buat class:

DrowsinessInference

Constructor:
- checkpoint_path
- device
- config_path

Saat initialization:
1. load config;
2. create model;
3. load checkpoint;
4. model.eval();
5. pindahkan model ke device.

Method:

predict(sequence)

Output:

{
    "not_drowsy_probability": float,
    "drowsy_probability": float,
    "prediction": int,
    "label": "NOT DROWSY" atau "DROWSY"
}

Threshold default:
0.50

Jangan melakukan threshold tuning di file ini.

Sanity check:
- probability 0–1
- probability sum sekitar 1
- tidak NaN

Jangan retrain.
Jangan membuat random weight.
WAJIB load checkpoint.
```

---

# 7. Prompt — `src/smoothing.py`

```text
BUAT FILE: src/smoothing.py

Tujuan:
Membuat prediksi realtime lebih stabil tanpa mengubah model.

Buat:

PredictionSmoother

Parameter:
window_size = 5

Input:
drowsy_probability

Simpan history probability.

Hitung:
moving average

Output:

{
    "raw_probability": float,
    "smoothed_probability": float,
    "prediction": int,
    "label": str
}

Aturan:
- DROWSY jika smoothed_probability >= threshold
- threshold default = 0.50

Jangan:
- mengubah checkpoint;
- training;
- mengubah EAR/MAR;
- menggunakan future frame.

Tambahkan reset().
```

---

# 8. Prompt — `src/realtime.py`

```text
BUAT FILE: src/realtime.py

Tujuan:
Membuat aplikasi realtime webcam menggunakan:

models/best_landmark_gru_ear_mar.pth

Pipeline WAJIB:

Webcam
  ->
MediaPipe Face Landmarker
  ->
facial landmarks
  ->
EAR V3 + MAR
  ->
sequence buffer 60 frame
  ->
training normalization
  ->
LandmarkGRU
  ->
probability
  ->
temporal smoothing
  ->
DROWSY / NOT DROWSY

============================================================
STARTUP CHECK
============================================================

Sebelum webcam:

1. load config
2. load checkpoint
3. buat model
4. dummy input (1,60,2)
5. inference
6. pastikan output valid
7. print:
   - device
   - parameter count
   - checkpoint path
   - input shape
   - output classes

Jika checkpoint gagal:
STOP.

Jangan membuat random model.

============================================================
WEBCAM
============================================================

Gunakan OpenCV.

Default:
camera index = 0

Support:

python -m src.realtime
python -m src.realtime --camera-index 0
python -m src.realtime --threshold 0.50
python -m src.realtime --smoothing-window 5

============================================================
FRAME PROCESSING
============================================================

Setiap frame:

1. capture
2. detect landmarks
3. validasi
4. extract EAR/MAR
5. jika valid:
   append ke buffer
6. jika buffer < 60:
   jangan prediksi
7. jika buffer = 60:
   inference
8. smoothing

============================================================
INVALID LANDMARK
============================================================

Jika:
- wajah tidak ditemukan
- landmark invalid
- EAR invalid
- MAR invalid

maka:

STATUS = UNKNOWN

Jangan memaksa:
DROWSY
atau
NOT DROWSY

Jangan memasukkan feature invalid ke model.

============================================================
OVERLAY
============================================================

Tampilkan:

FPS
Latency
EAR
MAR
Drowsy probability
Smoothed probability
STATUS
Landmark VALID/INVALID

Contoh:

FPS: 29.4
Latency: 25.2 ms

EAR: 0.284
MAR: 0.181

Drowsy probability: 0.72
Smoothed probability: 0.68

STATUS: DROWSY
Landmark: VALID

============================================================
VISUALIZATION
============================================================

Gambar:
- bounding box wajah
- landmark mata
- landmark mulut

Jangan membuat rendering terlalu berat.

============================================================
PERFORMANCE
============================================================

Hitung:
- FPS
- inference latency
- total frames
- valid landmark frames
- invalid landmark frames

============================================================
EXIT
============================================================

Tekan:
q

untuk keluar.

Pastikan:
camera.release()
cv2.destroyAllWindows()

============================================================
IMPORTANT
============================================================

Jangan:
- retrain
- mengubah architecture
- menggunakan relative EAR
- menggunakan PERCLOS
- menggunakan head pose
- menggunakan delta
- menghitung normalization dari webcam
- menggunakan random checkpoint

Model:
models/best_landmark_gru_ear_mar.pth
```

---

# 9. Prompt — `src/config.py`

```text
BUAT FILE: src/config.py

Tujuan:
Menjadi pusat konfigurasi aplikasi realtime.

Konfigurasi:

MODEL_PATH =
"models/best_landmark_gru_ear_mar.pth"

CONFIG_PATH =
"models/FINAL_landmark_gru_config.json"

MEDIAPIPE_MODEL_PATH =
"mediapipe/face_landmarker.task"

SEQUENCE_LENGTH = 60

THRESHOLD = 0.50

SMOOTHING_WINDOW = 5

CAMERA_INDEX = 0

CLASS_NAMES = {
    0: "NOT DROWSY",
    1: "DROWSY"
}

Jika config JSON menyimpan:
- mean
- std
- sequence_length
- architecture

gunakan JSON sebagai source of truth.

Jangan membuat nilai mean/std sendiri.
```

---

# 10. Prompt — `tests/test_model.py`

```text
BUAT FILE: tests/test_model.py

Tujuan:
Memastikan checkpoint kompatibel sebelum webcam dijalankan.

Test:

1. Load FINAL_landmark_gru_config.json
2. Load best_landmark_gru_ear_mar.pth
3. Create LandmarkGRU
4. Load state_dict
5. Dummy input:
   torch.randn(1,60,2)
6. inference
7. print:
   input shape
   output shape
   probabilities
   predicted class

Assertion:
- output shape == (1,2)
- probability bukan NaN
- probability >= 0
- probability <= 1
- probability sum sekitar 1

Jika gagal:
jelaskan penyebabnya.

Jangan retrain.
```

---

# 11. Prompt — `requirements.txt`

```text
BUAT FILE: requirements.txt

Dependency minimal:

- Python 3.10 atau 3.11
- PyTorch
- OpenCV
- MediaPipe Tasks API
- NumPy

Tambahkan dependency lain hanya jika benar-benar diperlukan.

Jangan memasukkan:
- TensorFlow jika tidak digunakan
- Ultralytics jika tidak digunakan
- library training yang tidak diperlukan

Pastikan dependency kompatibel dengan Windows.
```

---

# 12. Prompt — `README.md`

```text
BUAT FILE: README.md

Dokumentasikan:

1. Project overview

2. Model:
   Raw EAR + MAR
   LandmarkGRU
   2 GRU layers
   hidden=128
   temporal attention

3. Sequence:
   60 frames

4. Features:
   EAR
   MAR

5. Checkpoint:
   models/best_landmark_gru_ear_mar.pth

6. MediaPipe:
   mediapipe/face_landmarker.task

7. Installation:
   python -m venv .venv

Windows:
.venv\Scripts\activate

pip install -r requirements.txt

8. Run:
   python -m src.realtime

9. Optional:
   --camera-index
   --threshold
   --smoothing-window

10. Expected realtime output.

11. UNKNOWN state jika landmark invalid.

12. Catatan evaluasi:
   model ini adalah baseline terbaik yang teramati dalam eksperimen
   internal, bukan klaim terbaik secara universal.

13. Jangan mengklaim realtime accuracy tanpa dataset realtime
   dengan ground truth.
```

---

# 13. Struktur Akhir

Setelah semua prompt dijalankan, hasilnya harus:

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

# 14. Urutan Pengerjaan Agent

Jangan meminta agent membuat seluruh aplikasi sekaligus.

Urutan yang disarankan:

```text
1. model.py
       ↓
2. test_model.py
       ↓
3. landmark_detector.py
       ↓
4. feature_extractor.py
       ↓
5. preprocessing.py
       ↓
6. inference.py
       ↓
7. smoothing.py
       ↓
8. realtime.py
       ↓
9. requirements.txt
       ↓
10. README.md
```

**Prioritas tertinggi:** `model.py`, `feature_extractor.py`, dan `preprocessing.py`.

Ketiga file tersebut harus identik dengan pipeline training. Jangan sampai realtime memakai formula EAR atau normalization yang berbeda dari yang digunakan ketika `best_landmark_gru_ear_mar.pth` dilatih.
