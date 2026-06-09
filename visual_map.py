import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.fas_inference import FASDetector

model_path = "fas_model_efficientnet.pth"
detector = FASDetector(model_path, model_type="efficientnet")

def proses_visualisasi(image_path, label_nama):
    print(f"[PROSES] Membaca gambar: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"[ERROR] Gambar tidak ditemukan! Cek nama file: {image_path}")
        return

    label, conf, _ = detector.predict(frame)
    
    img = cv2.resize(frame, (224, 224))
    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(detector.device)
    
    with torch.no_grad():
        pred_maps, _ = detector.model(img_tensor)
        heatmap = pred_maps.squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (224, 224))

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')  
    
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet', alpha=0.7)
    plt.title(f"Prediction: {label} ({conf:.2f})")
    plt.axis('off')  
    
    plt.savefig(f"heatmap_{label_nama}.png", bbox_inches='tight', pad_inches=0)
    print(f"[BERHASIL] Heatmap disimpan sebagai: heatmap_{label_nama}.png")

if __name__ == "__main__":
    proses_visualisasi("wajah_asli.jpeg", "real")
    proses_visualisasi("wajah_spoof.jpeg", "spoof")