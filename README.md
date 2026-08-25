<div align="center">

# 🚛 FleetSense
**Multimodal Driver Risk Intelligence for Smart Logistics**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.x-61dafb.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.x-646CFF.svg)](https://vitejs.dev/)

*Theme: "AI for the Backbone of the Economy"*

</div>

---

## 📌 Executive Summary

**FleetSense** adalah sistem *Multimodal Driver Risk Intelligence* yang dirancang khusus untuk meningkatkan keselamatan dan efisiensi operasional armada logistik. Dengan menggabungkan tiga sudut penglihatan (kamera pengemudi, kabin, dan jalan), sistem ini mampu mendeteksi penurunan fokus, distraksi, serta risiko jalan secara simultan dan *real-time*.

Proyek ini dibangun untuk mengatasi tantangan kritis dalam **Smart Logistics**:
> *Mencegah kecelakaan berarti mencegah kerusakan kargo, keterlambatan pengiriman, dan hilangnya nyawa. Keandalan pengemudi adalah keandalan rantai pasok ekonomi.*

---

## ✨ Fitur Utama

- 👁️ **Spasio-Temporal Drowsiness Detection**
  Mendeteksi *microsleep* dan *yawning* berdasarkan sekuens temporal (*Facial Landmarks* 36-dimensi, MAR, EAR) menggunakan arsitektur kustom (CNN + BiGRU + Attention).
- 📱 **Distraction & Behavior Analysis**
  Mendeteksi 7 kelas perilaku pengemudi (seperti menggunakan HP, meraih barang ke belakang, minum) di dalam kabin menggunakan *ConvNeXt-Tiny*.
- 🛣️ **Road Context Awareness**
  Memantau kepadatan objek jalan (YOLOv8 Object Detection) dan menganalisis batas aman area kemudi (YOLOv8 Semantic Segmentation).
- 🧠 **XGBoost Decision Engine & Safety Policy**
  Menggabungkan 30 fitur persepsi dalam hitungan milidetik. Menghasilkan keputusan bertingkat (SAFE, CAUTION, HIGH, CRITICAL) menggunakan 8 model XGBoost independen.
- 🗣️ **Real-time Voice Telemetry & Warning**
  Dilengkapi dengan *Event Smoother* (Leaky Bucket) untuk mencegah *false alarms*, serta memberikan intervensi peringatan suara adaptif langsung ke pengemudi tanpa mengganggu fokus visual.
- 🛡️ **Graceful Degradation**
  Sistem dirancang *fault-tolerant*. Jika salah satu kamera terputus atau rusak tertutup debu, sistem otomatis beralih ke mode `DEGRADED` dan tetap beroperasi dengan sensor yang tersisa.

---

## 🏗️ Arsitektur Sistem

Alur kerja (Pipeline):
1. **Perception Layer**: Paralel ekstraksi frame (Mediapipe, ConvNeXt, YOLOv8).
2. **Feature Fusion**: Penyatuan skor probabilitas, fitur geometris, dan ketersediaan modalitas ke dalam satu Vektor Fitur.
3. **Decision Layer**: Eksekusi 8-model XGBoost Classifier + *Temporal Event Smoother*.
4. **Safety Policy**: Menilai kombinasi risiko (Misal: *Mengantuk* + *Risiko Jalan Tinggi* = **CRITICAL**).
5. **Action Layer**: Memicu peringatan suara melalui *Warning Manager* ke Web UI Frontend.

---

## 🚀 Instalasi & Cara Penggunaan

### Prasyarat
- Docker & Docker Compose
- Git LFS (Large File Storage) terinstal.
- GPU (NVIDIA CUDA) sangat disarankan untuk latensi *real-time*, meskipun CPU (XNNPACK) tetap didukung untuk *fallback*.

### Langkah Instalasi (via Docker)

1. **Clone repositori dan tarik model ML (Git LFS)**
   ```bash
   git clone https://github.com/username/fleetsense.git
   cd fleetsense
   git lfs install
   git lfs pull
   ```

2. **Jalankan layanan dengan Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Akses Dashboard Interaktif**
   Buka browser Anda dan tuju:
   👉 `http://localhost:5173` (Frontend Dashboard)
   👉 `http://localhost:8000/docs` (Backend API & Swagger UI)

---

## 📂 Struktur Proyek

```text
├── backend/
│   ├── models/            # Model ML terkompresi (.pth, .pt, .json)
│   ├── src/
│   │   ├── perception/    # Model spesifik (Drowsiness, StateFarm, Road)
│   │   ├── decision/      # XGBoost Decision Engine & Safety Policy
│   │   └── realtime/      # Realtime Pipeline, Smoother, & Recorder
│   ├── server.py          # FastAPI Server & WebSocket
│   └── Dockerfile
├── frontend/
│   ├── public/audio/      # File suara peringatan Bahasa Indonesia
│   ├── src/
│   │   ├── components/    # React Video/Dashboard components
│   │   └── services/      # WebSocket client & State management
│   ├── vite.config.ts
│   └── Dockerfile
├── docker-compose.yml     # Orkestrasi container
```

---

## 🧪 Evaluasi & Performa

* *Drowsiness Module*: Dievaluasi menggunakan subset dataset FL3D.
* *Behavior Module*: Terlatih pada dataset *State Farm Distracted Driver*.
* *Decision Engine*: Mampu melakukan fusi fitur dan menghasilkan *Severity Output* dalam latensi rata-rata **< 50ms** per frame, mencegah *bottleneck* pada proses persepsi video.



---

## 🛠️ Kontributor & Teknologi

Dibangun menggunakan stack teknologi modern untuk *AI Engineering* dan *Web Development*:
- **Backend/AI**: `Python`, `FastAPI`, `PyTorch`, `XGBoost`, `Ultralytics (YOLO)`, `MediaPipe`
- **Frontend**: `React`, `TypeScript`, `Vite`, `TailwindCSS`
- **Deployment**: `Docker`

<br>

<div align="center">
  <b>Membangun Logistik yang Lebih Aman dan Tangguh.</b><br>
  <i>Dipersembahkan untuk AIC Competition</i>
</div>
