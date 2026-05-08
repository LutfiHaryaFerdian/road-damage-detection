from ultralytics import YOLO
import torch

def main():

    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8s.pt")

    model.train(

        data="training/data.yaml",

        # ===== TRAINING =====
        epochs=100,
        imgsz=512,

        batch=4,          # TURUNKAN untuk GTX 1650
        workers=0,        # Windows lebih stabil

        device=0,

        name="road_damage_final",

        # ===== SPEED =====
        cache=True,
        amp=True,
        pretrained=True,

        # ===== AUGMENTATION =====
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        degrees=10,
        translate=0.1,
        scale=0.5,

        fliplr=0.5,
        mosaic=1.0,

        # ===== OPTIMIZER =====
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # ===== VALIDATION =====
        val=True,
        plots=True,

        # ===== EARLY STOP =====
        patience=15,

        save=True,
    )

if __name__ == "__main__":
    main()