import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from fas_inference import FASDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fas_model.pth")

# Jalur folder sudah dibuat dinamis agar aman di laptop siapa pun
BASE_TEST_DIR = os.path.join(BASE_DIR, "data", "Oulu-NPU")

def evaluate():
    print("="*50)
    print("   MEMULAI EVALUASI FACE ANTI-SPOOFING")
    print("="*50)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File {MODEL_PATH} tidak ditemukan!")
        return

    detector = FASDetector(MODEL_PATH)
    
    results = {
        "true": {"correct": 0, "total": 0, "no_face": 0},
        "false": {"correct": 0, "total": 0, "no_face": 0}
    }

    for label_name in ["true", "false"]:
        folder_path = os.path.join(BASE_TEST_DIR, label_name)
        
        print(f"\n[RADAR] Masuk ke folder: {folder_path}")
        if not os.path.exists(folder_path):
            print(f"[WARNING] Gagal! Folder {label_name} tidak ada.")
            continue
            
        semua_file = os.listdir(folder_path)
        
        file_valid = [f for f in semua_file if f.lower().endswith(('.jpg', '.jpeg', '.png', '.avi', '.mp4', '.mov'))]
        
        print(f"-> Total file ditemukan: {len(file_valid)}")

        for f_name in file_valid:
            file_path = os.path.join(folder_path, f_name)
            
            if f_name.lower().endswith(('.avi', '.mp4', '.mov')):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if not ret: continue
            else:
                frame = cv2.imread(file_path)
                if frame is None: continue

            
            label_pred, conf, bbox = detector.predict(frame)
            
            if label_pred == "NO_FACE":
                results[label_name]["no_face"] += 1
                continue

            results[label_name]["total"] += 1
            
            is_correct = False
            if label_name == "true" and label_pred == "REAL":
                is_correct = True
            elif label_name == "false" and label_pred == "SPOOF":
                is_correct = True
                
            if is_correct:
                results[label_name]["correct"] += 1

    print("\n" + "="*50)
    print("      HASIL AKHIR EVALUASI (OULU-NPU)")
    print("="*50)
    
    # --- PROSES PERHITUNGAN METRIK STANDAR ISO ---
    true_total = results["true"]["total"]
    true_correct = results["true"]["correct"]
    false_total = results["false"]["total"]
    false_correct = results["false"]["correct"]

    # Menghitung False Rejects (Wajah asli ditolak)
    false_rejects = true_total - true_correct
    # Menghitung False Accepts (Wajah palsu lolos)
    false_accepts = false_total - false_correct

    # Menghitung persentase
    bpcer = (false_rejects / true_total * 100) if true_total > 0 else 0.0 # BPCER
    apcer = (false_accepts / false_total * 100) if false_total > 0 else 0.0 # APCER
    acer = (bpcer + apcer) / 2 # ACER

    # Akurasi Umum
    total_all = true_total + false_total
    correct_all = true_correct + false_correct
    total_acc = (correct_all / total_all * 100) if total_all > 0 else 0

    print(f"Total File Diuji : {total_all}")
    print(f"Prediksi Benar   : {correct_all}")
    print(f"Akurasi Total    : {total_acc:.2f}%\n")
    
    print("-" * 50)
    print("METRIK FACE ANTI-SPOOFING (STANDAR ISO/IEC 30107-3):")
    print("-" * 50)
    print(f"1. BPCER (Salah menolak wajah asli) : {bpcer:.2f}%")
    print(f"2. APCER (Kebobolan wajah palsu)    : {apcer:.2f}%")
    print(f"3. ACER  (Rata-rata error)          : {acer:.2f}%")
    print("="*50)

    # --- VISUALISASI GRAFIK BAR ---
    print("[INFO] Menyimpan grafik evaluasi...")
    labels = ['BPCER\n(False Reject)', 'APCER\n(False Accept)', 'ACER\n(Average Error)']
    values = [bpcer, apcer, acer]
    colors = ['#FF9800', '#F44336', '#2196F3']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, max(values) + 15 if max(values) > 0 else 100)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.title('Face Anti-Spoofing Evaluation Metrics', fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Menambahkan angka persentase di atas batang grafik
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Simpan grafik ke file
    eval_plot_path = "fas_evaluation_metrics.png"
    plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] File gambar berhasil disimpan: {eval_plot_path}")
    
    # Tampilkan gambar di layar (opsional, bisa di-close kalau sudah muncul)
    plt.show()

if __name__ == "__main__":
    evaluate()