import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from fas_inference import FASDetector

def get_args():
    parser = argparse.ArgumentParser(description="Evaluasi FAS Multi-Model")
    parser.add_argument("--model_type", type=str, default="efficientnet", 
                        choices=["mobilenet", "shufflenet", "efficientnet"])
    return parser.parse_args()

def evaluate():
    args = get_args()
    
    print("="*50)
    print(f"   MEMULAI EVALUASI METRIK ISO: {args.model_type.upper()}")
    print("="*50)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file = os.path.join(BASE_DIR, f"fas_model_{args.model_type}.pth")

    # Proteksi ganda: Cek apakah file ada di root atau di dalam folder 'models'
    if not os.path.exists(model_file):
        model_file = os.path.join(BASE_DIR, "models", f"fas_model_{args.model_type}.pth")

    if not os.path.exists(model_file):
        print(f"[ERROR] File model '{model_file}' tidak ditemukan!")
        print("Pastikan hasil training DeepPixBis kamu sudah selesai dan filenya ada di direktori tersebut.")
        return

    # Inisialisasi Detector dengan arsitektur DeepPixBis baru kita
    detector = FASDetector(model_file, model_type=args.model_type)
    
    results = {
        "true": {"correct": 0, "total": 0},
        "false": {"correct": 0, "total": 0}
    }

    # Path ke Folder Dataset pengujian
    BASE_TEST_DIR = os.path.join(BASE_DIR, "data", "Oulu-NPU")

    for label_name in ["true", "false"]:
        folder_path = os.path.join(BASE_TEST_DIR, label_name)
        
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder {label_name} tidak ditemukan di {folder_path}")
            continue
            
        file_valid = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[INFO] Memproses {len(file_valid)} file di folder {label_name}...")

        for f_name in file_valid:
            file_path = os.path.join(folder_path, f_name)
            frame = cv2.imread(file_path)
            if frame is None: continue
            
            # Melakukan prediksi berbasis peta piksel lokal
            label_pred, conf, _ = detector.predict(frame)
            results[label_name]["total"] += 1
            
            # Cek ketepatan prediksi
            if label_name == "true" and label_pred == "REAL":
                results[label_name]["correct"] += 1
            elif label_name == "false" and label_pred == "SPOOF":
                results[label_name]["correct"] += 1

    # Hitung Metrik ISO/IEC 30107-3
    true_total = results["true"]["total"]
    true_correct = results["true"]["correct"]
    false_total = results["false"]["total"]
    false_correct = results["false"]["correct"]

    # BPCER: Salah menolak wajah asli
    bpcer = ((true_total - true_correct) / true_total * 100) if true_total > 0 else 0
    # APCER: Kebobolan wajah palsu
    apcer = ((false_total - false_correct) / false_total * 100) if false_total > 0 else 0
    # ACER: Rata-rata Error total
    acer = (bpcer + apcer) / 2

    print("\n" + "="*50)
    print("           HASIL AKHIR EVALUASI (DEEPPIXBIS MODEL)")
    print("="*50)
    print(f"Model Yang Diuji : {args.model_type.upper()}")
    print(f"Total Data Uji   : {true_total + false_total}")
    print("-" * 50)
    print(f"1. BPCER (False Reject) : {bpcer:.2f}%")
    print(f"2. APCER (False Accept) : {apcer:.2f}%")
    print(f"3. ACER  (Average Error): {acer:.2f}%")
    print("="*50)

    # Simpan Grafik Batang baru
    plt.figure(figsize=(8, 5))
    labels = ['BPCER', 'APCER', 'ACER']
    values = [bpcer, apcer, acer]
    plt.bar(labels, values, color=['#FF9800', '#F44336', '#2196F3'])
    plt.ylabel('Error Rate (%)')
    plt.title(f'Evaluation Metrics - {args.model_type.upper()} (DeepPixBis)')
    
    plot_path = f"evaluation_metrics_{args.model_type}.png"
    plt.savefig(plot_path)
    print(f"[INFO] Grafik evaluasi baru berhasil disimpan: {plot_path}")

if __name__ == "__main__":
    evaluate()