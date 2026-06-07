import matplotlib.pyplot as plt

# Data Epoch
epochs = list(range(1, 16))

# Data Loss asli dari hasil log terminal Hasun semalam
loss_mobilenet = [0.5903, 0.5192, 0.5030, 0.4876, 0.4746, 0.4647, 0.4545, 0.4407, 0.4399, 0.4286, 0.4229, 0.4193, 0.4117, 0.4069, 0.4005]
loss_shufflenet = [0.6663, 0.6300, 0.6034, 0.5839, 0.5695, 0.5594, 0.5508, 0.5436, 0.5393, 0.5352, 0.5319, 0.5288, 0.5265, 0.5241, 0.5226]
loss_efficientnet = [0.6303, 0.5962, 0.5721, 0.5548, 0.5384, 0.5323, 0.5162, 0.5038, 0.4985, 0.4912, 0.4874, 0.4808, 0.4681, 0.4671, 0.4626]

# Menggambar Grafik
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_efficientnet, marker='^', label='EfficientNet-B0 (Loss)', color='#2196F3', linewidth=2)
plt.plot(epochs, loss_mobilenet, marker='o', label='MobileNetV2 (Loss)', color='#FF9800', linewidth=2)
plt.plot(epochs, loss_shufflenet, marker='s', label='ShuffleNetV2 (Loss)', color='#F44336', linewidth=2)

# Format Grafik
plt.title('Kurva Training Loss - 15 Epoch (Metode DeepPixBis)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Tingkat Kesalahan)', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Simpan Grafik
plt.savefig('training_loss_comparison.png', dpi=300, bbox_inches='tight')
print("[SUKSES] Grafik Training Loss berhasil disulap dan disimpan sebagai 'training_loss_comparison.png'")