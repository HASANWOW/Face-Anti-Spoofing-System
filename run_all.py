import os
import time

# Daftar model yang akan dilatih dan dievaluasi otomatis
# Efficientnet tidak dimasukkan karena semalam sudah selesai.
models_to_run = ["mobilenet", "shufflenet", "efficientnet"]

print("="*60)
print(" 🚀 MEMULAI PROSES AUTO-TRAINING & EVALUATION 🚀")
print("="*60)

for model in models_to_run:
    print("\n" + "*"*50)
    print(f"🔥 FASE 1: TRAINING MODEL {model.upper()} (15 EPOCH) 🔥")
    print("*"*50)
    
    # Menjalankan perintah training
    os.system(f"python src/train_deeppixbis.py --model_type {model} --epochs 15")
    
    # Jeda 5 detik agar RAM/VRAM laptop bernapas sebentar
    time.sleep(5)
    
    print("\n" + "*"*50)
    print(f"📊 FASE 2: EVALUASI & CETAK GRAFIK {model.upper()} 📊")
    print("*"*50)
    
    # Menjalankan perintah evaluasi
    os.system(f"python src/evaluate.py --model_type {model}")
    
    # Jeda 5 detik sebelum pindah ke model berikutnya
    time.sleep(5)

print("\n" + "="*60)
print(" 🎉 ALHAMDULILLAH, SEMUA PROSES SELESAI DENGAN SUKSES! 🎉")
print(" Silakan cek folder untuk melihat file .pth dan grafik .png")
print("="*60)