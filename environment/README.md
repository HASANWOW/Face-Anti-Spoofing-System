# Research FAS — Face Anti-Spoofing + Face Recognition
---
## STRUKTUR PROJECT LENGKAP 

```
## 🛡️ Two-Layer Face Anti-Spoofing & Recognition System

## 📌 Deskripsi Proyek
Proyek ini adalah sistem keamanan berbasis *Computer Vision* yang menggunakan arsitektur keamanan dua lapis. Sistem ini tidak hanya mengenali identitas seseorang secara *real-time* melalui kamera, tetapi juga memiliki pertahanan aktif terhadap serangan manipulasi wajah (*Presentation Attack*) seperti penggunaan foto cetak atau layar ponsel.

Sistem ini dikembangkan sebagai bagian dari proyek penelitian untuk menguji ketahanan model deteksi *liveness* wajah menggunakan dataset standar internasional.
--
## ⚙️ Arsitektur Sistem (Pipeline)

Sistem bekerja dalam hitungan milidetik dengan alur sebagai berikut:
1. **Lapis Pertama (Face Anti-Spoofing):** Menggunakan model klasifikasi (Trained Model) untuk menganalisis tekstur wajah dan membedakan antara benda 3D (Manusia Asli) dan benda 2D (Foto/Layar HP). Jika terdeteksi sebagai benda mati (SPOOF), akses langsung diblokir.

2. **Lapis Kedua (Face Recognition):** Jika wajah tervalidasi sebagai manusia asli (REAL), sistem menggunakan algoritma *FaceNet/InsightFace* untuk mengekstrak vektor wajah dan mencocokkannya dengan *database* karyawan/mahasiswa menggunakan metrik *Cosine Distance*.
--
## 📂 Struktur Direktori
*Catatan: Direktori data dataset dan bobot model tidak disertakan di repositori ini karena batasan ukuran file.*

```text
├── data/                  # (Git-ignored) Dataset OULU-NPU untuk training & evaluasi
├── dataset_wajah/         # (Git-ignored) Folder identitas wajah (contoh: /edwin, /hasan)
├── models/                # (Git-ignored) Tempat menyimpan fas_model.pth & .pkl
├── src/
│   ├── train_fas.py       # Skrip untuk melatih model Anti-Spoofing
│   ├── evaluate.py        # Skrip pengujian akurasi menggunakan OULU-NPU
│   ├── code_recognition.py# Ekstraksi *embeddings* wajah ke database (.pkl)
│   └── main.py            # Aplikasi Inference Real-Time menggunakan Webcam
├── .gitignore
└── README.md
"""