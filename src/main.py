import torch
import cv2
import time
import numpy as np
import pickle
import os
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
from fas_inference import FASDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAS_MODEL_PATH = os.path.join(BASE_DIR, "fas_model_efficientnet.pth") 
RECOG_MODEL_PATH = os.path.join(BASE_DIR, "models", "model_recognition_arcface.pkl")

COSINE_THRESHOLD = 0.75                  # Threshold pengenalan wajah ArcFace
FRAME_SKIP       = 3                     # Proses AI setiap 3 frame agar tidak patah-patah
FAS_CONFIDENCE_THRESHOLD = 0.50          # Batas keyakinan detektor FAS

print("=" * 50)
print("  Memuat model Face Anti-Spoofing (EFFICIENTNET V2)...")
print("=" * 50)

if not os.path.exists(FAS_MODEL_PATH):
    FAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "fas_model_efficientnet.pth")

if not os.path.exists(FAS_MODEL_PATH):
    print(f"\n[ERROR] File '{FAS_MODEL_PATH}' tidak ditemukan!")
    exit()

# Inisialisasi mesin pendeteksi dengan target model kita
fas_detector = FASDetector(FAS_MODEL_PATH, model_type="efficientnet")

print("\n  Memuat database Face Recognition ArcFace...")
if not os.path.exists(RECOG_MODEL_PATH):
    print(f"\n[ERROR] File '{RECOG_MODEL_PATH}' tidak ditemukan!")
    exit()

with open(RECOG_MODEL_PATH, "rb") as f:
    data = pickle.loads(f.read())
known_encodings = data["encodings"]
known_names     = data["names"]
print(f"  [V] {len(known_names)} wajah dimuat dari database.")

print("\n  Memuat InsightFace untuk pencarian wajah...")
insight_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
insight_app.prepare(ctx_id=0, det_size=(320, 320))
print("  [V] InsightFace siap.")

print("\n  Membuka Kamera Utama...")
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

print("\n" + "=" * 50)
print("  SISTEM AKTIF — Tekan 'Q' untuk keluar")
print("=" * 50 + "\n")

frame_count     = 0
last_results    = []   

def recognize_face(embedding):
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
    for res in results:
        x, y, w, h   = res["bbox_xywh"]
        fas_label    = res["fas_label"]
        fas_conf     = res["fas_conf"]
        person_name  = res["name"]
        recog_dist   = res["distance"]

        if fas_label == "REAL":
            color = (0, 255, 0) if person_name != "Tidak Dikenal" else (0, 165, 255)
        elif fas_label == "SPOOF":
            color = (0, 0, 255)
        else:
            color = (128, 128, 128)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        fas_text = f"FAS: {fas_label} ({fas_conf:.0%})"
        cv2.rectangle(frame, (x, y - 28), (x + w, y), color, cv2.FILLED)
        cv2.putText(frame, fas_text, (x + 4, y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

        if fas_label == "REAL":
            name_text = f"{person_name} ({recog_dist:.2f})"
            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 28), color, cv2.FILLED)
            cv2.putText(frame, name_text, (x + 4, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
        elif fas_label == "SPOOF":
            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 28), color, cv2.FILLED)
            cv2.putText(frame, "!! SPOOFING DETECTED !!", (x + 4, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Frame tidak terbaca dari kamera.")
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        current_results = []
        
        # SINKRONISASI: Biarkan InsightFace mencari lokasi wajah terlebih dahulu
        faces = insight_app.get(frame)

        if not faces:
            last_results = []
        else:
            face = faces[0]
            bx = face.bbox.astype(int)
            x = max(0, bx[0])
            y = max(0, bx[1])
            w = max(0, bx[2] - bx[0])
            h = max(0, bx[3] - bx[1])
            
            # Crop wajah untuk diprediksi tekstur liveness-nya
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size > 0:
                # Kirim frame potongan ke model EfficientNet
                fas_label, fas_conf, _ = fas_detector.predict(face_crop)
                
                if fas_label == "REAL" and fas_conf >= FAS_CONFIDENCE_THRESHOLD:
                    # Jalankan pencocokan nama jika wajah terbukti REAL
                    name, dist = recognize_face(face.embedding)
                    current_results.append({
                        "bbox_xywh": (x, y, w, h),
                        "fas_label": "REAL",
                        "fas_conf":  fas_conf,
                        "name":      name,
                        "distance":  dist,
                    })
                else:
                    # Jika terdeteksi SPOOF, langsung potong/blokir jalur recognition demi keamanan
                    current_results.append({
                        "bbox_xywh": (x, y, w, h),
                        "fas_label": "SPOOF",
                        "fas_conf":  fas_conf,
                        "name":      "",
                        "distance":  1.0,
                    })
            else:
                last_results = []

        last_results = current_results

    frame = draw_results(frame, last_results)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("FAS + Face Recognition | Tekan Q untuk keluar", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nSistem dihentikan.")