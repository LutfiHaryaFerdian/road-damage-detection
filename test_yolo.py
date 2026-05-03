from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Predict
results = model("https://ultralytics.com/images/bus.jpg")

# Ambil hasil gambar dengan bounding box
annotated_frame = results[0].plot()

# Simpan pakai OpenCV (lebih reliable)
cv2.imwrite("hasil.jpg", annotated_frame)

print("Deteksi berhasil, cek file hasil.jpg")