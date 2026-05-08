from ultralytics import YOLO

model = YOLO("runs/detect/road_damage_final/weights/best.pt")

results = model.predict(
    source="dataset/test/images",
    save=True,
    conf=0.25
)

print("Selesai test")