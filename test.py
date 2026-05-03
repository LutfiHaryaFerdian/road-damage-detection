from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source="dataset/test/images",
    save=True,
    conf=0.25
)

print("Selesai test")