from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="training/data.yaml",
    epochs=1,
    imgsz=640
)