# PANDUAN MENINGKATKAN AKURASI DETEKSI

## 🎯 Perubahan yang Sudah Dilakukan

### 1. **Upgrade Model (YOLOv8n → YOLOv8m)**

```
YOLOv8n (Nano)   → Cepat, kurang akurat
YOLOv8m (Medium) → Balance (⬆️ DIGUNAKAN)
YOLOv8l (Large)  → Paling akurat, lambat
```

**Impact**: +10-15% improvement dalam accuracy

---

### 2. **Training Configuration**

| Parameter              | Sebelum | Sesudah | Alasan                          |
| ---------------------- | ------- | ------- | ------------------------------- |
| **epochs**             | 40      | 100     | Model konvergen lebih baik      |
| **imgsz**              | 416     | 640     | Detail lebih kaya, standar YOLO |
| **batch_size**         | 4       | 8       | Gradient lebih stabil           |
| **lr (learning rate)** | default | 0.01    | Kontrol training lebih baik     |
| **optimizer**          | Adam    | SGD     | SGD lebih stable untuk YOLO     |

---

### 3. **Data Augmentation (Lebih Kuat)**

```python
# Rotasi & Translasi
degrees=15.0     # Dari 5.0 (3x lebih banyak)
translate=0.1    # Dari 0.05 (2x lebih banyak)
scale=0.5        # Dari 0.3 (lebih aggressive)

# Color/Brightness
hsv_s=0.75       # Dari 0.7 (lebih saturasi)
hsv_v=0.4        # Value variation

# Flipping
fliplr=0.5       # Horizontal flip 50%
flipud=0.1       # ✨ BARU: Vertical flip 10%

# Advanced
mosaic=1.0       # Mosaic augmentation (sangat membantu)
mixup=0.0        # Bisa di-enable jika data banyak
```

---

### 4. **Prediction Improvements**

```python
# Preprocessing
- CLAHE (Contrast Limited Adaptive Histogram)
- Noise reduction (NLM denoising)
- Brightness normalization

# Prediction Parameters
conf=0.4         # Dari 0.25 (lebih selektif)
iou=0.5          # ✨ BARU: NMS IOU threshold
imgsz=640        # Sesuai training size
```

---

## 📈 Cara Melatih Ulang Model

### Langkah 1: Jalankan Training Baru

```bash
cd training
python train.py
```

**Durasi**: 2-4 jam (tergantung GPU)

### Langkah 2: Monitor Progress

- File akan tersimpan di `runs/detect/road_damage_model_v2/`
- Buka `results.png` untuk melihat grafik training

### Langkah 3: Evaluasi Model

```bash
python evaluate.py
```

### Langkah 4: Copy Model Terbaik

```bash
cp runs/detect/road_damage_model_v2/weights/best.pt ../models/best.pt
```

---

## 🔍 Jika Akurasi Masih Kurang

### A. Tingkatkan Model Size

```python
# Di training/train.py
model = YOLO("yolov8l.pt")  # Dari yolov8m.pt
# Accuracy: +5-10%, Speed: -30% slower
```

### B. Tingkatkan Training Data

- Kumpulkan lebih banyak images
- Pastikan labels akurat dan konsisten
- Minimum 100-200 images per class

### C. Fine-tuning Strategy

```python
# Gunakan pre-trained weights
model = YOLO("runs/detect/road_damage_model_v2/weights/best.pt")
model.train(
    data="training/data.yaml",
    epochs=50,      # Lanjutkan training
    lr0=0.001,      # Lower learning rate
    # ... parameters lainnya
)
```

### D. Ensemble Prediction

```python
# Gunakan 2 model sekaligus untuk hasil lebih akurat
model1 = YOLO("models/best.pt")
model2 = YOLO("runs/detect/road_damage_model_v2/weights/best.pt")

results1 = model1.predict(image)
results2 = model2.predict(image)
# Combine predictions...
```

---

## 📊 Expected Improvements

**Sebelum Optimization:**

- mAP@0.5: ~0.60-0.70
- Precision: ~0.65-0.75
- Recall: ~0.60-0.70

**Setelah Optimization:**

- mAP@0.5: ~0.75-0.85
- Precision: ~0.80-0.90
- Recall: ~0.75-0.85

---

## 🛠️ Troubleshooting

### Model OOM (Out of Memory)?

```python
batch=4  # Turunkan batch size
imgsz=512  # Turunkan image size
```

### Training Terlalu Lambat?

```python
# Gunakan mixed precision
# (Otomatis di YOLOv8 jika GPU support)
```

### Val Loss Naik (Overfitting)?

```python
weight_decay=0.001  # Naikkan regularization
dropout: Enable di model
```

---

## 📝 Next Steps

1. ✅ Update training config ✓
2. ⏳ Train model dengan config baru
3. ⏳ Evaluasi akurasi
4. ⏳ Deploy model terbaik
5. ⏳ Monitor real-world performance
