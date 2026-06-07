import os
import io
import cv2
import time
import random
import argparse
import tarfile
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="Training FAS dengan OULU-NPU (.tar)")
    parser.add_argument("--train_tar",  type=str, required=True,
                        help="Path ke file Train_files.tar")
    parser.add_argument("--output",     type=str, default="fas_model.pth")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--frames",     type=int, default=8,
                        help="Jumlah frame diambil per video")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Batasi jumlah video (0=semua). Pakai 200-300 jika RAM terbatas")
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Proporsi data validasi (default 20%)")
    parser.add_argument("--lr",         type=float, default=0.001)
    
    # [MODIFIKASI] Tombol Saklar Model
    parser.add_argument("--model",      type=str, default="mobilenet", 
                        choices=["mobilenet", "shufflenet", "efficientnet"],
                        help="Pilih arsitektur model lightweight yang ingin dilatih")
    return parser.parse_args()

def get_label_from_name(video_name):
    base = os.path.splitext(os.path.basename(video_name))[0]
    parts = base.split("_")
    if len(parts) < 4:
        return None
    attack_type = parts[-1]
    return 1 if attack_type == "1" else 0


def parse_bbox_txt(txt_bytes):
    bboxes = {}
    try:
        content = txt_bytes.decode("utf-8")
        for line in content.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) == 5:
                idx = int(parts[0])
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                bboxes[idx] = (x1, y1, x2, y2)
    except Exception:
        pass
    return bboxes


def extract_frames_from_bytes(video_bytes, bbox_dict, num_frames=8):
    tmp_path = f"/tmp/_oulu_tmp_{random.randint(0,999999)}.avi"
    try:
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return []

        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames  = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]

            if bbox_dict:
                closest = min(bbox_dict.keys(), key=lambda k: abs(k - idx))
                x1, y1, x2, y2 = bbox_dict[closest]

                pad = int(0.1 * min(x2 - x1, y2 - y1))
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                if x2 > x1 and y2 > y1:
                    face = frame[y1:y2, x1:x2]
                else:
                    face = frame
            else:
                face = frame

            face_resized = cv2.resize(face, (224, 224))
            frames.append(face_resized)

        cap.release()
        return frames

    except Exception as e:
        return []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def scan_tar(tar_path, max_videos=0):
    print(f"\n  Membuka file .tar: {tar_path}")
    print(f"  (Proses ini mungkin butuh 10-30 detik pertama kali...)\n")

    tar = tarfile.open(tar_path, "r")
    members = tar.getmembers()

    avi_members = {m.name: m for m in members if m.name.endswith(".avi")}
    txt_members = {m.name: m for m in members if m.name.endswith(".txt")}

    print(f"  Ditemukan {len(avi_members)} file video (.avi)")
    print(f"  Ditemukan {len(txt_members)} file bbox  (.txt)")

    samples = []
    for avi_name, avi_member in avi_members.items():
        label = get_label_from_name(avi_name)
        if label is None:
            continue

        txt_name = avi_name.replace(".avi", ".txt")
        txt_member = txt_members.get(txt_name, None)

        samples.append((avi_name, label, txt_name if txt_member else None))

    n_real  = sum(1 for s in samples if s[1] == 1)
    n_spoof = sum(1 for s in samples if s[1] == 0)
    print(f"\n  Label: {n_real} REAL | {n_spoof} SPOOF | Total: {len(samples)}")

    if max_videos > 0 and len(samples) > max_videos:
        random.shuffle(samples)
        real_samples  = [s for s in samples if s[1] == 1][:max_videos // 2]
        spoof_samples = [s for s in samples if s[1] == 0][:max_videos // 2]
        samples = real_samples + spoof_samples
        random.shuffle(samples)
        print(f"  Dibatasi ke {len(samples)} video (--max_videos {max_videos})")

    return samples, tar


class OULUTarDataset(Dataset):
    def __init__(self, samples, tar_path, num_frames=8, transform=None):
        self.samples    = samples
        self.tar_path   = tar_path
        self.num_frames = num_frames
        self.transform  = transform
        self._tar       = None

    def _get_tar(self):
        if self._tar is None:
            self._tar = tarfile.open(self.tar_path, "r")
        return self._tar

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        avi_name, label, txt_name = self.samples[idx]
        tar = self._get_tar()

        try:
            avi_member  = tar.getmember(avi_name)
            avi_file    = tar.extractfile(avi_member)
            video_bytes = avi_file.read()
        except Exception:
            dummy = torch.zeros(3, 224, 224)
            return dummy, torch.tensor(label, dtype=torch.long)

        bbox_dict = {}
        if txt_name:
            try:
                txt_member = tar.getmember(txt_name)
                txt_file   = tar.extractfile(txt_member)
                bbox_dict  = parse_bbox_txt(txt_file.read())
            except Exception:
                pass

        frames = extract_frames_from_bytes(video_bytes, bbox_dict, self.num_frames)

        if not frames:
            dummy = torch.zeros(3, 224, 224)
            return dummy, torch.tensor(label, dtype=torch.long)

        frame = random.choice(frames)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# [MODIFIKASI] Pabrik Model dengan 3 Pilihan Lightweight CNN
def build_model(model_name, num_classes=2):
    if model_name == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Model {model_name} tidak dikenali!")

    # Hitung Ukuran Parameter untuk laporan
    total_params = sum(p.numel() for p in model.parameters())
    print("-" * 55)
    print(f"[*] Arsitektur Terpilih : {model_name.upper()}")
    print(f"[*] Total Parameter     : {total_params / 1e6:.2f} Juta Parameter")
    print("-" * 55)
    
    return model


def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    from tqdm import tqdm # Tambahan import tqdm yang terlewat
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0
    mode_str   = "Train" if is_train else "Val"

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc=f"  {mode_str}", leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 55)
    print("  OULU-NPU Face Anti-Spoofing Training")
    print("=" * 55)
    print(f"  Device     : {device}")
    print(f"  Train .tar : {args.train_tar}")
    print(f"  Model      : {args.model.upper()}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Frames/vid : {args.frames}")
    if args.max_videos > 0:
        print(f"  Max videos : {args.max_videos}")
    print("=" * 55)

    all_samples, _ = scan_tar(args.train_tar, max_videos=args.max_videos)

    if len(all_samples) == 0:
        print("\n[ERROR] Tidak ada sampel ditemukan. Cek path --train_tar")
        return

    # Mencegah error validasi pada dataset sangat kecil
    val_size    = max(1, int(len(all_samples) * args.val_split))
    val_samples = all_samples[:val_size]
    trn_samples = all_samples[val_size:]
    print(f"\n  Split → Train: {len(trn_samples)} | Val: {len(val_samples)}")

    #  Transformasi 
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    trn_ds = OULUTarDataset(trn_samples, args.train_tar, args.frames, train_tf)
    val_ds = OULUTarDataset(val_samples, args.train_tar, args.frames, val_tf)

    trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # [MODIFIKASI] Memanggil Pabrik Model
    model = build_model(args.model).to(device)
    
    # [MODIFIKASI] Mengembalikan Class Weight ke posisi Netral karena data diseimbangkan
    weights = torch.tensor([1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    print(f"\n  Mulai training {args.epochs} epoch...\n")
    best_val_acc = 0.0
    best_epoch   = 0

    history_train_loss = []
    history_val_loss = []
    history_train_acc = []
    history_val_acc = []

    # Dinamis Output Name
    final_output_path = args.output if args.output != "fas_model.pth" else f"fas_model_{args.model}.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{args.epochs}")

        trn_loss, trn_acc = run_epoch(model, trn_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)

        history_train_loss.append(trn_loss)
        history_val_loss.append(val_loss)
        history_train_acc.append(trn_acc)
        history_val_acc.append(val_acc)

        elapsed = time.time() - t0
        print(f"  Train → Loss: {trn_loss:.4f} | Acc: {trn_acc:.2f}%")
        print(f"  Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"  Waktu : {elapsed:.1f} detik")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          val_acc,
                "train_acc":        trn_acc,
                "protocol":         f"Model {args.model.upper()} - OULU-NPU",
            }, final_output_path)
            print(f"  ✅ Model terbaik disimpan! Val Acc = {val_acc:.2f}%")
        print()

    print("=" * 55)
    print(f"  Training selesai!")
    print(f"  Best: Epoch {best_epoch} → Val Acc = {best_val_acc:.2f}%")
    print(f"  Model disimpan di: {final_output_path}")
    print("=" * 55)

    print("\n[INFO] Sedang membuat grafik performa training (Academic Style)...")
    
    epochs_range = range(1, args.epochs + 1)
    
    plt.rcParams.update({
        "font.family": "serif",        
        "font.serif": ["Times New Roman"], 
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs_range, history_train_acc, label='Training Acc', color='black', linestyle='-', marker='o', markersize=4)
    ax1.plot(epochs_range, history_val_acc, label='Validation Acc', color='dimgray', linestyle='--', marker='s', markersize=4)
    ax1.set_title(f'Accuracy ({args.model.upper()})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend(loc='lower right', frameon=False) 
    ax1.grid(True, linestyle=':', alpha=0.6)     

    ax2.plot(epochs_range, history_train_loss, label='Training Loss', color='black', linestyle='-', marker='o', markersize=4)
    ax2.plot(epochs_range, history_val_loss, label='Validation Loss', color='dimgray', linestyle='--', marker='s', markersize=4)
    ax2.set_title(f'Loss ({args.model.upper()})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.legend(loc='upper right', frameon=False)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    # Dinamis Plot Name
    save_path_plot = f"training_metrics_{args.model}.png"
    plt.savefig(save_path_plot, dpi=300, bbox_inches='tight')
    print(f"[INFO] Sukses! Grafik standar paper telah disimpan di: {save_path_plot}")
    
    print("\nLangkah selanjutnya:")
    print("  python src/evaluate.py")
    print("  python src/main.py")

if __name__ == "__main__":
    main()