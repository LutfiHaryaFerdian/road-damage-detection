from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os
import cv2
import numpy as np

app = FastAPI()

model = YOLO("models/best.pt")

UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def preprocess_image(image_path):
    """Preprocessing untuk meningkatkan akurasi deteksi"""
    img = cv2.imread(image_path)
    
    # Normalize brightness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Noise reduction
    img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21)
    
    cv2.imwrite(image_path, img)
    return image_path


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # simpan file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Preprocessing untuk meningkatkan akurasi
    file_path = preprocess_image(file_path)

    # prediksi dengan confidence threshold lebih optimal
    results = model.predict(
        source=file_path,
        conf=0.4,           # ⬆️ NAIKKAN dari 0.25 untuk precision lebih baik
        iou=0.5,            # ⬆️ TAMBAHKAN IOU threshold (NMS)
        save=True,
        project=RESULT_DIR,
        name="predict",
        imgsz=640,          # ⬆️ MATCH dengan training size
        max_det=100         # Max detections per image
    )

    # ambil path hasil yang BENAR
    save_dir = results[0].save_dir

    # cari file gambar hasil
    files = [f for f in os.listdir(save_dir) if f.endswith((".jpg", ".png"))]

    if not files:
        return {"error": "Tidak ada hasil gambar"}

    output_image = os.path.join(save_dir, files[0])

    return FileResponse(output_image)