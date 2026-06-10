import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import cv2
import time
import numpy as np
import pickle
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
from src.fas_inference import FASDetector 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAS_MODEL_PATH = os.path.join(BASE_DIR, "fas_model_efficientnet.pth") 
RECOG_MODEL_PATH = os.path.join(BASE_DIR, "models", "model_recognition_arcface.pkl")
COSINE_THRESHOLD = 0.75
FRAME_SKIP = 3
FAS_CONFIDENCE_THRESHOLD = 0.49

print("[INFO] Memuat model FAS (EfficientNet) & ArcFace...")
if not os.path.exists(FAS_MODEL_PATH):
    print(f"[ERROR] Model FAS tidak ditemukan di: {FAS_MODEL_PATH}")
    exit()

fas_detector = FASDetector(FAS_MODEL_PATH, model_type="efficientnet")
insight_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
insight_app.prepare(ctx_id=0, det_size=(320, 320))

with open(RECOG_MODEL_PATH, "rb") as f:
    data = pickle.loads(f.read())
known_encodings = data["encodings"]
known_names = data["names"]

cap = cv2.VideoCapture(0)

print("\n" + "="*50)
print(" SISTEM AKTIF — Tekan 'Q' untuk keluar")
print("="*50 + "\n")

frame_count = 0
last_results = []

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    if frame_count % FRAME_SKIP == 0:
        faces = insight_app.get(frame)
        current_results = []
        
        if faces:
            face = faces[0]
            bx = face.bbox.astype(int)
            x, y, w, h = max(0, bx[0]), max(0, bx[1]), max(0, bx[2]-bx[0]), max(0, bx[3]-bx[1])
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size > 0:
                fas_label, fas_conf, _ = fas_detector.predict(face_crop)
                
                if fas_label == "REAL" and fas_conf >= FAS_CONFIDENCE_THRESHOLD:
                    min_dist = 1.0
                    name = "Tidak Dikenal"
                    
                    for i, known_enc in enumerate(known_encodings):
                        dist = cosine(face.embedding, known_enc)
                        if dist < min_dist:
                            min_dist = dist
                            if min_dist < COSINE_THRESHOLD:
                                name = known_names[i]
                    
                    current_results.append({"bbox": (x,y,w,h), "fas_label": "REAL", "fas_conf": fas_conf, "name": name, "dist": min_dist})
                else:
                    current_results.append({"bbox": (x,y,w,h), "fas_label": "SPOOF", "fas_conf": fas_conf, "name": "", "dist": 1.0})
        
        last_results = current_results

    for res in last_results:
        x, y, w, h = res["bbox"]
        color = (0, 255, 0) if res["fas_label"] == "REAL" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{res['fas_label']} ({res['fas_conf']:.0%})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if res["fas_label"] == "REAL":
            cv2.putText(frame, f"{res['name']} ({res['dist']:.2f})", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("FAS + Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()