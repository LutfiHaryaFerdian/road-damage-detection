from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

model = YOLO("models/best.pt")

UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # simpan file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # prediksi + simpan hasil ke folder custom
    results = model.predict(
        source=file_path,
        conf=0.25,
        save=True,
        project=RESULT_DIR,
        name="predict"
    )

    # ambil path hasil yang BENAR
    save_dir = results[0].save_dir

    # cari file gambar hasil
    files = [f for f in os.listdir(save_dir) if f.endswith((".jpg", ".png"))]

    if not files:
        return {"error": "Tidak ada hasil gambar"}

    output_image = os.path.join(save_dir, files[0])

    return FileResponse(output_image)