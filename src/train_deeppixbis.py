import os
import tarfile
import random
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

def get_args():
    parser = argparse.ArgumentParser(description="Training DeepPixBis 3 Model")
    parser.add_argument("--model_type", type=str, default="efficientnet", choices=["mobilenet", "shufflenet", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()

class DeepPixBisModel(nn.Module):
    def __init__(self, model_type="efficientnet"):
        super(DeepPixBisModel, self).__init__()
        self.model_type = model_type
        
        if model_type == "efficientnet":
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.features = backbone.features
            in_channels = 1280
        elif model_type == "mobilenet":
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.features = backbone.features
            in_channels = 1280
        elif model_type == "shufflenet":
            backbone = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
            self.features = nn.Sequential(
                backbone.conv1, backbone.maxpool, backbone.stage2, 
                backbone.stage3, backbone.stage4, backbone.conv5
            )
            in_channels = 1024
            
        for param in self.features.parameters():
            param.requires_grad = False
            
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

class OULUDeepPixDataset(Dataset):
    def __init__(self, samples, tar_path, transform=None):
        self.samples = samples
        self.tar_path = tar_path
        self.transform = transform
        self._tar = None

    def __getitem__(self, idx):
        if self._tar is None: self._tar = tarfile.open(self.tar_path, "r")
        name, label = self.samples[idx]
        try:
            v_bytes = self._tar.extractfile(self._tar.getmember(name)).read()
            tmp_path = f"tmp_{random.randint(0,99999)}.avi"
            with open(tmp_path, "wb") as f: f.write(v_bytes)
            cap = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, max(0, total-1)))
            ret, frame = cap.read()
            cap.release()
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
            img = cv2.resize(frame, (224, 224))
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if self.transform: img = self.transform(img)
            
            target_map = torch.ones((1, 14, 14)) if label == 1 else torch.zeros((1, 14, 14))
            return img, target_map, torch.tensor([label], dtype=torch.float32)
        except:
            return torch.zeros(3,224,224), torch.zeros((1, 14, 14)), torch.tensor([label], dtype=torch.float32)

    def __len__(self): return len(self.samples)

def scan_tar(tar_path):
    tar = tarfile.open(tar_path, "r")
    avi_members = [m.name for m in tar.getmembers() if m.name.endswith(".avi")]
    samples = []
    for name in avi_members:
        base = os.path.splitext(os.path.basename(name))[0]
        parts = base.split("_")
        if len(parts) >= 4:
            label = 1 if parts[-1] == "1" else 0
            samples.append((name, label))
    return samples

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tar_path = "data/Train_files (1).tar"
    
    if not os.path.exists(tar_path):
        print(f"[ERROR] File tidak ditemukan: {tar_path}")
        return

    all_samples = scan_tar(tar_path)
    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.8)
    
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(OULUDeepPixDataset(all_samples[:split], tar_path, tf), batch_size=16, shuffle=True)
    
    print(f"\n[START] Memulai Training Model: {args.model_type.upper()}")
    model = DeepPixBisModel(model_type=args.model_type).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total Parameter Model: {total_params:,} | Parameter Dilatih: {train_params:,}")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0
        for imgs, maps, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, maps, labels = imgs.to(device), maps.to(device), labels.to(device)
            optimizer.zero_grad()
            
            pred_maps, pred_scores = model(imgs)
            loss_pixel = criterion(pred_maps, maps)
            loss_charts = criterion(pred_scores, labels)
            loss = 0.5 * loss_pixel + 0.5 * loss_charts
            
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            
        print(f"-> Epoch {epoch} Selesai | Loss: {loss_total/len(train_loader):.4f}")
        
    out_name = f"fas_model_{args.model_type}.pth"
    torch.save(model.state_dict(), out_name)
    print(f"\n[SUKSES] Model disimpan sebagai: {out_name}")

if __name__ == "__main__":
    main()