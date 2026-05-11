import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os
import pickle

app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset_wajah")
OUTPUT_PKL = os.path.join(BASE_DIR, "models", "model_recognition_arcface.pkl")

known_face_encodings = []
known_face_names = []


for person_name in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person_name)
    
    if os.path.isdir(person_dir):
        print(f"Memproses wajah: {person_name}")
        
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpeg', '.png', '.jpg')): 
                image_path = os.path.join(person_dir, filename)
                
                try:
                    img = cv2.imread(image_path)
                    
                    if img is None:
                        print(f"Peringatan: Gagal memuat gambar di {image_path}. Lewati.")
                        continue

                    faces = app.get(img)

                    if len(faces) == 1:
                        known_face_encodings.append(faces[0].embedding)
                        known_face_names.append(person_name)
                        print(f" [V] Berhasil ekstrak: {filename}")
                    elif len(faces) > 1:
                        print(f"Peringatan: Lebih dari satu wajah terdeteksi di {image_path}. Lewati.")
                    else:
                        print(f"Wajah tidak terdeteksi di {image_path}. Lewati.")
                        
                except Exception as e:
                    print(f"Gagal memproses {image_path}: {e}")

data = {"encodings": np.array(known_face_encodings), "names": known_face_names}

with open(OUTPUT_PKL, "wb") as f:
    f.write(pickle.dumps(data))

print(f"Data encoding absensi telah disimpan di: {OUTPUT_PKL}")