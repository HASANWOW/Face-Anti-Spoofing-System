import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.fas_inference import FASDetector
from insightface.app import FaceAnalysis

model_path = "fas_model_efficientnet.pth"
detector = FASDetector(model_path, model_type="efficientnet")

print("[INFO] Memuat modul InsightFace...")
insight_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
insight_app.prepare(ctx_id=0, det_size=(320, 320))

def proses_visualisasi(image_path, label_nama):
    print(f"\n[PROSES] Membaca gambar: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"[ERROR] Gambar tidak ditemukan! Cek nama file: {image_path}")
        return

    faces = insight_app.get(frame)
    if len(faces) == 0:
        print(f"[ERROR] Tidak ada wajah yang terdeteksi di {image_path}!")
        return
        
    face = faces[0]
    bx = face.bbox.astype(int)
    x, y, w, h = max(0, bx[0]), max(0, bx[1]), max(0, bx[2]-bx[0]), max(0, bx[3]-bx[1])
    face_crop = frame[y:y+h, x:x+w]

    label, conf, _ = detector.predict(face_crop)
    
    img = cv2.resize(face_crop, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_float = img_rgb.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_float - mean) / std
    
    img_tensor = torch.from_numpy(img_norm.transpose((2, 0, 1))).unsqueeze(0).to(detector.device).float()
    
    with torch.no_grad():
        pred_maps, _ = detector.model(img_tensor)
        heatmap = pred_maps.squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (224, 224))

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Input")
    plt.axis('off')  
    
    plt.subplot(1, 2, 2)

    
    plt.imshow(heatmap, cmap='jet', alpha=0.7, vmin=0.1, vmax=1)
    plt.title(f"Prediction: {label} ({conf:.2f})")
    plt.axis('off')  
    
    plt.savefig(f"heatmap_{label_nama}.png", bbox_inches='tight', pad_inches=0)
    print(f"[BERHASIL] Heatmap disimpan sebagai: heatmap_{label_nama}.png")

if __name__ == "__main__":
    proses_visualisasi("wajah_asli.jpeg", "real")
    proses_visualisasi("wajah_spoof.jpeg","spoof")
   