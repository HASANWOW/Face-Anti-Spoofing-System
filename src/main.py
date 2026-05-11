import cv2
import time
import numpy as np
import pickle
import os
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
from fas_inference import FASDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "fas_model.pth")
RECOG_MODEL_PATH = os.path.join(BASE_DIR, "models", "model_recognition_arcface.pkl")
COSINE_THRESHOLD = 0.75                        # Threshold pengenalan wajah
FRAME_SKIP       = 3                              # Proses AI setiap N frame (hemat CPU)
FAS_CONFIDENCE_THRESHOLD = 0.50                  # Minimal keyakinan FAS (50%)



# 1. LOAD MODEL FAS
print("=" * 50)
print("  Memuat model Face Anti-Spoofing...")
print("=" * 50)

if not os.path.exists(FAS_MODEL_PATH):
    print(f"\n[ERROR] File '{FAS_MODEL_PATH}' tidak ditemukan!")
    print("  Jalankan dulu: python train_fas.py --oulu_root ... --video_root ...")
    exit()

fas_detector = FASDetector(FAS_MODEL_PATH)

# 2. LOAD DATABASE FACE RECOGNITION

print("\n  Memuat database Face Recognition...")

if not os.path.exists(RECOG_MODEL_PATH):
    print(f"\n[ERROR] File '{RECOG_MODEL_PATH}' tidak ditemukan!")
    print("  Jalankan dulu: python code_recognition.py")
    exit()

with open(RECOG_MODEL_PATH, "rb") as f:
    data = pickle.loads(f.read())
known_encodings = data["encodings"]
known_names     = data["names"]
print(f"  [V] {len(known_names)} wajah dimuat dari database.")

# 3. INISIALISASI INSIGHTFACE (untuk recognition)

print("\n  Memuat InsightFace untuk recognition...")
insight_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
insight_app.prepare(ctx_id=0, det_size=(320, 320))
print("  [V] InsightFace siap.")

# 4. BUKA KAMERA

print("\n  Membuka kamera...")
cap = None
for index in [0, 1, 0 + cv2.CAP_DSHOW]:
    temp = cv2.VideoCapture(index)
    time.sleep(0.5)
    if temp.isOpened():
        cap = temp
        print(f"  [V] Kamera terbuka di indeks {index}.")
        break
    temp.release()

if cap is None:
    print("[ERROR] Kamera tidak bisa dibuka!")
    exit()

# 5. LOOP UTAMA

print("\n" + "=" * 50)
print("  SISTEM AKTIF — Tekan 'Q' untuk keluar")
print("=" * 50 + "\n")

frame_count     = 0
last_results    = []   


def recognize_face(embedding):
    """Cocokkan embedding dengan database, kembalikan nama + jarak."""
    name         = "Tidak Dikenal"
    min_distance = 1.0
    for i, known_enc in enumerate(known_encodings):
        dist = cosine(embedding, known_enc)
        if dist < min_distance:
            min_distance = dist
            if min_distance < COSINE_THRESHOLD:
                name = known_names[i]
    return name, min_distance


def draw_results(frame, results):
    """Gambar kotak, label FAS, dan nama orang di frame."""
    for res in results:
        x, y, w, h   = res["bbox_xywh"]
        fas_label    = res["fas_label"]
        fas_conf     = res["fas_conf"]
        person_name  = res["name"]
        recog_dist   = res["distance"]

        # Warna kotak: Hijau = real & dikenal, Oranye = real & tidak dikenal,
        #              Merah = spoof, Abu = no_face
        if fas_label == "REAL":
            color = (0, 255, 0) if person_name != "Tidak Dikenal" else (0, 165, 255)
        elif fas_label == "SPOOF":
            color = (0, 0, 255)
        else:
            color = (128, 128, 128)

        # Kotak wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label FAS (atas kotak)
        fas_text = f"FAS: {fas_label} ({fas_conf:.0%})"
        cv2.rectangle(frame, (x, y - 28), (x + w, y), color, cv2.FILLED)
        cv2.putText(frame, fas_text, (x + 4, y - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

        # Label nama (bawah kotak) — hanya tampil jika REAL
        if fas_label == "REAL":
            name_text = f"{person_name} ({recog_dist:.2f})"
            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 28), color, cv2.FILLED)
            cv2.putText(frame, name_text, (x + 4, y + h + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
        elif fas_label == "SPOOF":
            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 28), color, cv2.FILLED)
            cv2.putText(frame, "!! SPOOFING DETECTED !!", (x + 4, y + h + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Frame tidak terbaca dari kamera.")
        break

    frame_count += 1

    # ── Proses AI setiap FRAME_SKIP frame ──
    if frame_count % FRAME_SKIP == 0:
        current_results = []

        # Jalankan FAS terlebih dahulu
        fas_label, fas_conf, bbox = fas_detector.predict(frame)

        if fas_label == "NO_FACE":
            # Tidak ada wajah — tidak perlu proses recognition
            last_results = []
        else:
            x, y, w, h = bbox

            if fas_label == "REAL" and fas_conf >= FAS_CONFIDENCE_THRESHOLD:
                # Wajah asli dan cukup yakin → jalankan recognition
                faces = insight_app.get(frame)
                if faces:
                    # Ambil wajah yang posisinya paling dekat dengan bbox FAS
                    face        = faces[0]
                    name, dist  = recognize_face(face.embedding)
                    # Konversi bbox InsightFace (x1,y1,x2,y2) ke (x,y,w,h)
                    bx = face.bbox.astype(int)
                    rx = max(0, bx[0])
                    ry = max(0, bx[1])
                    rw = max(0, bx[2] - bx[0])
                    rh = max(0, bx[3] - bx[1])
                    current_results.append({
                        "bbox_xywh": (rx, ry, rw, rh),
                        "fas_label": "REAL",
                        "fas_conf":  fas_conf,
                        "name":      name,
                        "distance":  dist,
                    })
                else:
                    current_results.append({
                        "bbox_xywh": (x, y, w, h),
                        "fas_label": "REAL",
                        "fas_conf":  fas_conf,
                        "name":      "Tidak Dikenal",
                        "distance":  1.0,
                    })
            else:
                # Spoof atau confidence terlalu rendah → blokir
                current_results.append({
                    "bbox_xywh": (x, y, w, h),
                    "fas_label": fas_label,
                    "fas_conf":  fas_conf,
                    "name":      "",
                    "distance":  1.0,
                })

            last_results = current_results

    #  Gambar hasil di frame (setiap frame) 
    frame = draw_results(frame, last_results)

    # Tampilkan FPS sederhana
    cv2.putText(frame, f"Frame: {frame_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("FAS + Face Recognition | Tekan Q untuk keluar", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


#  Bersihkan 
cap.release()
cv2.destroyAllWindows()
print("\nSistem dihentikan.")