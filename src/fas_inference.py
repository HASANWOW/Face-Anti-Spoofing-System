import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models



# Transformasi untuk inference (sama seperti val_transform di training)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def _build_model(num_classes=2):
    """Bangun arsitektur MobileNetV2 (sama persis dengan saat training)."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class FASDetector:
    """
    Kelas utama untuk deteksi Face Anti-Spoofing secara real-time.
    
    Cara pakai:
        detector = FASDetector("fas_model.pth")
        label, confidence = detector.predict(frame_bgr)
        # label: "REAL" atau "SPOOF"
        # confidence: angka 0.0–1.0 (seberapa yakin modelnya)
    """
    
    def __init__(self, model_path, device=None):
        """
        Args:
            model_path : path ke file .pth hasil training
            device     : "cuda" atau "cpu". Auto-detect jika None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = _build_model(num_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()  
        
        # Detector wajah OpenCV (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        print(f"[FAS] Model berhasil diload dari {model_path}")
        print(f"[FAS] Protocol: {checkpoint.get('protocol', '?')} | "
              f"Best Val Acc: {checkpoint.get('val_acc', 0):.2f}%")
        print(f"[FAS] Device: {self.device}")
    
    def predict(self, frame_bgr):
        """
        Prediksi apakah wajah di frame adalah REAL atau SPOOF.
        
        Args:
            frame_bgr : frame dari webcam dalam format BGR (OpenCV)
        
        Returns:
            label      : "REAL" atau "SPOOF"
            confidence : float 0.0–1.0 (keyakinan prediksi)
            bbox       : (x, y, w, h) kotak wajah yang dideteksi, atau None
        """
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        
        if len(faces) == 0:
            # Tidak ada wajah — tidak bisa memprediksi
            return "NO_FACE", 0.0, None
        
        # Ambil wajah terbesar
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Crop wajah dengan sedikit padding
        pad = int(0.1 * min(w, h))
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame_bgr.shape[1], x + w + pad)
        y2  = min(frame_bgr.shape[0], y + h + pad)
        face_crop = frame_bgr[y1:y2, x1:x2]
        
        # Konversi BGR → RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Terapkan transformasi
        tensor = INFERENCE_TRANSFORM(face_rgb).unsqueeze(0).to(self.device)
        
        # Prediksi
        with torch.no_grad():
            output = self.model(tensor)
            probs  = torch.softmax(output, dim=1)
            pred   = probs.argmax(dim=1).item()
            conf   = probs[0][pred].item()
        
        # Indeks 1 = real, 0 = spoof (sesuai label di training)
        label = "REAL" if pred == 1 else "SPOOF"
        return label, conf, (x, y, w, h)