# AIC — SafeRoute AI

> **AI-Powered Driver & Road Safety Analysis for Logistics Fleets**

SafeRoute AI adalah prototipe sistem AI untuk meningkatkan keandalan dan efisiensi operasional armada logistik dengan mendeteksi peningkatan risiko selama proses pengiriman. Sistem menganalisis kondisi pengemudi dan lingkungan jalan secara bersamaan untuk mengidentifikasi situasi yang berpotensi menyebabkan kecelakaan, sehingga gangguan seperti kerusakan barang, keterlambatan pengiriman, dan downtime kendaraan dapat diminimalkan melalui peringatan dini kepada pengemudi.

Sistem menganalisis video perjalanan dengan menggabungkan:

* Kondisi pengemudi
* Kondisi jalan
* Pergerakan objek
* Urutan kejadian dari waktu ke waktu
* Konteks keseluruhan situasi
* Tingkat risiko

Ketika risiko mencapai level tertentu, sistem memberikan **visual alert** dan **audio alert** kepada pengemudi.

---

## 1. Product Concept

SafeRoute AI tidak hanya melakukan object detection.

Sistem mencoba menjawab pertanyaan:

> **"Apakah kombinasi kondisi pengemudi dan kondisi jalan saat ini cukup berbahaya untuk memberikan peringatan?"**

Contoh:

```text
Driver menggunakan HP
        +
Kendaraan depan semakin dekat
        +
Kendaraan depan melakukan pengereman
        +
Motor mendekat
        ↓
   RISK INCREASED
        ↓
   ALERT DRIVER
```

Dengan pendekatan tersebut, sistem diharapkan dapat mengurangi **false alarm** dan menghindari peringatan yang terlalu sering.

---

# 2. MVP Scope

Versi MVP menggunakan aplikasi berbasis web.

### Input

Satu video perjalanan kendaraan yang merepresentasikan:

* Driver-facing camera
* Road-facing camera

### Driver Analysis

Sistem menganalisis indikasi:

* Driver distraction
* Penggunaan smartphone
* Driver melihat ke arah lain
* Indikasi kantuk
* Kehilangan perhatian terhadap jalan

### Road Analysis

Sistem menganalisis:

* Kendaraan
* Sepeda motor
* Pejalan kaki
* Kendaraan di depan
* Pergerakan objek
* Potensi pengereman
* Perubahan jarak objek

### Context Analysis

Sistem menggabungkan hasil analisis:

```text
Driver State
     +
Road State
     +
Temporal Information
     +
Object Movement
     ↓
Context Analysis
     ↓
Risk Assessment
```

### Output

Sistem menghasilkan:

* Risk level
* Risk reason
* Visual warning
* Audio alert

---

# 3. Risk Level

SafeRoute AI menggunakan beberapa level risiko.

```text
┌─────────────────────────────────────┐
│             RISK LEVEL              │
├─────────────────────────────────────┤
│                                     │
│  LOW                                │
│  Tidak ada kondisi berbahaya        │
│                                     │
│  POTENTIAL                          │
│  Kondisi mulai menunjukkan risiko   │
│                                     │
│  WARNING                            │
│  Risiko cukup signifikan            │
│                                     │
│  CRITICAL                           │
│  Risiko kecelakaan sangat tinggi    │
│                                     │
└─────────────────────────────────────┘
```

Contoh:

```text
LOW
↓
Tidak ada alert


POTENTIAL
↓
Monitoring kondisi


WARNING
↓
"BIP"
"Perhatikan jalan."


CRITICAL
↓
"BIP-BIP-BIP"
"Bahaya. Risiko tabrakan."
```

---

# 4. Alert Strategy

Sistem tidak boleh memberikan alarm setiap kali sebuah objek terdeteksi.

Untuk menghindari **alert fatigue**, digunakan beberapa mekanisme.

### Persistence

Kondisi harus bertahan dalam periode tertentu sebelum dianggap signifikan.

### Context

Satu deteksi tidak langsung menghasilkan alarm.

Contoh:

```text
Smartphone detected
        ↓
Tidak langsung alarm
        ↓
Driver distraction duration
        +
Road condition
        ↓
Risk Assessment
```

### Cooldown

Setelah alarm diberikan, sistem tidak langsung mengulang alarm yang sama.

### Escalation

Alarm meningkat ketika tingkat risiko meningkat.

```text
Potential
    ↓
Warning
    ↓
Critical
```

Tujuannya:

> **SafeRoute AI lebih memilih tidak mengganggu pengemudi daripada memberikan alarm yang tidak diperlukan.**

---

# 5. System Architecture

```text
                        ┌─────────────────────┐
                        │       USER          │
                        │                     │
                        │ Upload Video        │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │      FRONTEND       │
                        │   React + Vite      │
                        │                     │
                        │ • Video Upload      │
                        │ • Video Player      │
                        │ • Risk Display      │
                        │ • Alert Status      │
                        └──────────┬──────────┘
                                   │
                              HTTP / REST
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │       BACKEND       │
                        │       Flask         │
                        │                     │
                        │  API & Orchestration│
                        └──────────┬──────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
          ┌────────────┐   ┌────────────┐   ┌────────────┐
          │   Driver   │   │    Road    │   │  Temporal  │
          │  Analysis  │   │  Analysis  │   │  Analysis  │
          └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
                │                │                │
                └────────────────┼────────────────┘
                                 ▼
                        ┌─────────────────────┐
                        │  Context Analysis   │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │    Risk Engine      │
                        │                     │
                        │ Risk Assessment     │
                        │ Risk Scoring        │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Alert Manager     │
                        │                     │
                        │ Visual + Audio      │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │      FRONTEND       │
                        │                     │
                        │ Risk Result         │
                        │ Alert Status        │
                        └─────────────────────┘
```

---

# 6. Technology Stack

## Frontend

| Technology | Purpose           |
| ---------- | ----------------- |
| React      | User Interface    |
| Vite       | Frontend tooling  |
| JavaScript | Application logic |

## Backend

| Technology | Purpose              |
| ---------- | -------------------- |
| Python     | AI processing        |
| Flask      | REST API             |
| OpenCV     | Video processing     |
| NumPy      | Numerical processing |
| YOLO       | Object detection     |

## Infrastructure

| Technology     | Purpose                      |
| -------------- | ---------------------------- |
| Docker         | Application containerization |
| Docker Compose | Local service orchestration  |
| Git            | Version control              |

---

# 7. Fixed Project Structure

Struktur ini menjadi **struktur folder utama AIC**.

```text
aic/
│
├── frontend/
│   ├── public/
│   │
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoUpload.jsx
│   │   │   ├── VideoPlayer.jsx
│   │   │   ├── RiskResult.jsx
│   │   │   ├── RiskIndicator.jsx
│   │   │   └── AlertStatus.jsx
│   │   │
│   │   ├── pages/
│   │   │   └── Analysis.jsx
│   │   │
│   │   ├── services/
│   │   │   └── api.js
│   │   │
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   │
│   ├── package.json
│   ├── vite.config.js
│   ├── Dockerfile
│   └── .dockerignore
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   │
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   └── analysis.py
│   │   │
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── driver.py
│   │   │   ├── road.py
│   │   │   ├── temporal.py
│   │   │   └── context.py
│   │   │
│   │   ├── risk/
│   │   │   ├── __init__.py
│   │   │   ├── risk_engine.py
│   │   │   └── alert_manager.py
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── video_service.py
│   │   │   └── analysis_service.py
│   │   │
│   │   └── core/
│   │       ├── __init__.py
│   │       └── config.py
│   │
│   ├── models/
│   │   └── .gitkeep
│   │
│   ├── tests/
│   │   └── .gitkeep
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
│
├── videos/
│   ├── input/
│   │   └── .gitkeep
│   │
│   └── output/
│       └── .gitkeep
│
├── docs/
│   └── ARCHITECTURE.md
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

# 8. Folder Responsibilities

## `frontend/`

Seluruh aplikasi user interface.

```text
frontend/
└── src/
```

### `components/`

Komponen UI yang dapat digunakan kembali.

```text
VideoUpload.jsx
VideoPlayer.jsx
RiskResult.jsx
RiskIndicator.jsx
AlertStatus.jsx
```

### `pages/`

Halaman aplikasi.

```text
Analysis.jsx
```

### `services/`

Komunikasi dengan Flask API.

```text
api.js
```

---

# 9. Backend Architecture

Backend menggunakan pendekatan modular.

## `routes/`

Menangani HTTP request.

```text
routes/
├── health.py
└── analysis.py
```

Contoh endpoint:

```text
GET  /api/health
POST /api/analysis
GET  /api/analysis/<id>
```

---

## `analysis/`

Tempat seluruh proses computer vision dan AI analysis.

### `driver.py`

Menganalisis kondisi pengemudi.

Contoh:

```text
Driver distraction
Phone usage
Head direction
Drowsiness indication
Attention level
```

### `road.py`

Menganalisis lingkungan jalan.

Contoh:

```text
Vehicle
Motorcycle
Pedestrian
Vehicle movement
Object distance
Traffic condition
```

### `temporal.py`

Menganalisis perubahan kondisi berdasarkan waktu.

Contoh:

```text
Frame 1 → vehicle normal
Frame 2 → vehicle closer
Frame 3 → vehicle braking
Frame 4 → critical distance
```

### `context.py`

Menggabungkan hasil driver analysis dan road analysis.

```text
Driver State
     +
Road State
     +
Temporal State
     ↓
Context
```

---

# 10. Risk Engine

Folder:

```text
backend/app/risk/
```

## `risk_engine.py`

Bertanggung jawab menentukan tingkat risiko.

Contoh konsep:

```text
Driver Distraction       +20
Phone Usage              +25
Vehicle Too Close        +20
Vehicle Braking          +20
Motorcycle Approaching   +15
--------------------------------
Total                    100
```

Kemudian:

```text
0 - 20     → LOW
21 - 40    → POTENTIAL
41 - 70    → WARNING
71 - 100   → CRITICAL
```

> Nilai tersebut merupakan konsep awal dan dapat dikalibrasi berdasarkan hasil eksperimen.

---

# 11. Alert Manager

File:

```text
backend/app/risk/alert_manager.py
```

Bertanggung jawab menentukan kapan sistem harus memberikan alert.

Logika:

```text
Risk Level
    ↓
Check Persistence
    ↓
Check Context
    ↓
Check Cooldown
    ↓
Check Previous Alert
    ↓
Generate Alert
```

Output dapat berupa:

```text
NONE

WARNING
"Perhatikan jalan."

CRITICAL
"Bahaya. Risiko tabrakan."

DROWSINESS
"Anda terlihat mengantuk."
```

---

# 12. Services

Folder:

```text
backend/app/services/
```

## `video_service.py`

Bertanggung jawab menangani:

* Upload video
* Validasi file
* Penyimpanan video
* Video metadata
* Frame extraction

## `analysis_service.py`

Bertindak sebagai orchestrator proses AI.

```text
Video
  ↓
Driver Analysis
  ↓
Road Analysis
  ↓
Temporal Analysis
  ↓
Context Analysis
  ↓
Risk Engine
  ↓
Alert Manager
  ↓
Result
```

---

# 13. Models

Folder:

```text
backend/models/
```

Digunakan untuk menyimpan model AI.

Contoh:

```text
backend/models/
├── yolo_model.pt
├── driver_model.pt
└── ...
```

Model AI tidak dimasukkan ke Git repository apabila ukurannya besar.

---

# 14. Video Processing

```text
videos/
├── input/
└── output/
```

### `input/`

Video yang diberikan oleh pengguna.

### `output/`

Video hasil processing apabila MVP membutuhkan visualisasi bounding box atau event.

Contoh:

```text
input/
└── truck_trip_01.mp4

output/
└── truck_trip_01_analyzed.mp4
```

---

# 15. Docker Architecture

AIC menggunakan Docker untuk menjalankan aplikasi secara konsisten.

```text
                 Docker Compose
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        React Container    Flask Container
           :5173               :5000
              │                 │
              └────── HTTP ─────┘
```

### Frontend

```text
React + Vite
```

### Backend

```text
Flask + Python + AI
```

Run seluruh aplikasi:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

---

# 16. Development Workflow

Development dilakukan secara bertahap.

```text
1. Project Foundation
        ↓
2. Flask API
        ↓
3. React UI
        ↓
4. React ↔ Flask
        ↓
5. Video Upload
        ↓
6. Video Processing
        ↓
7. Driver Analysis
        ↓
8. Road Analysis
        ↓
9. Temporal Analysis
        ↓
10. Context Analysis
        ↓
11. Risk Engine
        ↓
12. Alert Manager
        ↓
13. Audio Alert
        ↓
14. End-to-End Demo
```

---

# 17. MVP Development Principles

AIC mengikuti beberapa prinsip:

### 1. Build Simple First

Jangan membuat infrastruktur yang belum diperlukan.

### 2. AI Must Be Explainable

Setiap risk level harus memiliki alasan.

Contoh:

```text
CRITICAL RISK

Reasons:
- Driver distraction detected
- Vehicle ahead braking
- Motorcycle approaching
```

### 3. Context Over Detection

Object detection bukan output akhir.

```text
Detection
    ↓
Interpretation
    ↓
Context
    ↓
Risk
```

### 4. Avoid Alert Fatigue

Sistem tidak boleh memberikan alarm secara berlebihan.

### 5. Modular Architecture

Setiap bagian AI dapat dikembangkan secara independen.

---

# 18. Current MVP Limitations

Versi MVP **tidak mencakup**:

* Fleet management
* User authentication
* GPS tracking
* Database histori perjalanan
* Mobile application
* IoT device
* Hardware camera
* Automatic vehicle control
* Autonomous driving
* Automatic braking
* Cloud infrastructure

Fokus MVP adalah:

> **Video → AI Analysis → Risk Assessment → Alert**

---

# 19. Future Development

Setelah MVP berhasil, sistem dapat dikembangkan menjadi platform keselamatan armada.

```text
Vehicle Camera
      ↓
Edge AI Device
      ↓
SafeRoute AI
      ↓
Real-time Risk Detection
      ↓
Driver Alert
      ↓
Fleet Monitoring Platform
      ↓
Historical Safety Analytics
```

Pengembangan berikutnya dapat mencakup:

* Real-time camera processing
* Edge AI
* Fleet dashboard
* Driver safety score
* Trip history
* GPS integration
* Cloud processing
* Event streaming
* Notification system
* Fleet analytics
* Multi-vehicle monitoring

---

# 20. Project Goal

Tujuan utama AIC bukan sekadar membuat model computer vision yang mampu mendeteksi objek.

Tujuan project adalah membangun sistem yang mampu memahami:

```text
"What is happening?"
        ↓
"Is it becoming dangerous?"
        ↓
"How serious is the risk?"
        ↓
"Should the driver be alerted?"
```

Dengan demikian:

> **SafeRoute AI bukan hanya mendeteksi bahaya. SafeRoute AI mencoba mengenali kapan sebuah situasi mulai berubah menjadi risiko dan memberikan peringatan pada saat yang tepat.**

---

# 21. Project Status

**Current Phase: Foundation**

```text
[✓] Project initialized
[✓] React installed
[✓] Flask installed
[✓] Docker installed
[ ] Docker Compose configuration
[ ] Flask health API
[ ] React ↔ Flask connection
[ ] Video upload
[ ] Driver analysis
[ ] Road analysis
[ ] Temporal analysis
[ ] Context analysis
[ ] Risk engine
[ ] Alert manager
[ ] Audio alert
[ ] End-to-end MVP
```

---

# 22. Quick Start

## Development without Docker

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
source .venv/bin/activate
python app.py
```

---

## Development with Docker

From project root:

```bash
docker compose up --build
```

Stop containers:

```bash
docker compose down
```

View logs:

```bash
docker compose logs -f
```

---

# 23. Repository Structure Summary

```text
aic/
│
├── frontend/        → React application
│
├── backend/         → Flask + AI engine
│
├── videos/          → Input/output video
│
├── docs/            → Architecture documentation
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## SafeRoute AI

**Detect → Understand → Assess → Alert**

> Building safer roads through contextual AI.
