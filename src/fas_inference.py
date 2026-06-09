import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np

class DeepPixBisModel(nn.Module):
    def __init__(self, model_type="efficientnet"):
        super(DeepPixBisModel, self).__init__()
        self.model_type = model_type
        
        if model_type == "efficientnet":
            backbone = models.efficientnet_b0(weights=None)
            self.features = backbone.features
            in_channels = 1280
        elif model_type == "mobilenet":
            backbone = models.mobilenet_v2(weights=None)
            self.features = backbone.features
            in_channels = 1280
        elif model_type == "shufflenet":
            backbone = models.shufflenet_v2_x1_0(weights=None)
            self.features = nn.Sequential(
                backbone.conv1, backbone.maxpool, backbone.stage2, 
                backbone.stage3, backbone.stage4, backbone.conv5
            )
            in_channels = 1024
            
        self.pixel_conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False)
        self.score_fc = nn.Sequential(
            nn.Linear(1 * 14 * 14, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        pixel_map = self.pixel_conv(x)
        pixel_map = self.upsample(pixel_map)
        pixel_map = torch.sigmoid(pixel_map)
        flat_map = pixel_map.view(pixel_map.size(0), -1)
        global_score = torch.mean(flat_map, dim=1, keepdim=True)
        return pixel_map, global_score

class FASDetector:
    def __init__(self, model_path, model_type="efficientnet"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        print(f"[INFO] Mengaktifkan Arsitektur DeepPixBis - {model_type.upper()}...")
        self.model = DeepPixBisModel(model_type=model_type)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def predict(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose((2, 0, 1))
        
        img = torch.from_numpy(img).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            pred_maps, pred_scores = self.model(img)
            conf = pred_scores.item()
            label = "REAL" if conf >= 0.3 else "SPOOF"
            if label == "SPOOF":
                conf = 1.0 - conf
            
        return label, conf, None