from ultralytics import YOLO
import torch

def main():
    print("CUDA:", torch.cuda.is_available())

    model = YOLO("yolov8n.pt")

    model.train(
        data="training/data.yaml",
        epochs=40,
        imgsz=416,

        batch=4,        # 🔥 TURUNKAN
        workers=0,      # 🔥 WAJIB di Windows
        device=0,

        name="road_damage_model",

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5.0,
        translate=0.05,
        scale=0.3,

        fliplr=0.5,
        flipud=0.0
    )

if __name__ == "__main__":
    main()