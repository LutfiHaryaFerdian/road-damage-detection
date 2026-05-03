"""
Script untuk evaluasi akurasi model pada test set
Menampilkan metrik: Precision, Recall, mAP@0.5, mAP@0.5:0.95
"""

from ultralytics import YOLO
import json
from pathlib import Path

def evaluate_model(model_path="models/best.pt"):
    """Evaluasi model pada test set"""
    
    print("=" * 60)
    print("EVALUASI MODEL DETEKSI KERUSAKAN JALAN")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # Validasi pada test set
    results = model.val(
        data="training/data.yaml",
        imgsz=640,
        conf=0.4,
        iou=0.5,
        batch=8,
        device=0
    )
    
    # Print hasil
    print("\n📊 HASIL EVALUASI:")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    # Per-class results
    print("\n📋 HASIL PER KELAS:")
    class_names = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
    for i, name in enumerate(class_names):
        print(f"  {name}:")
        print(f"    - Precision: {results.box.mp_per_class[i] if i < len(results.box.mp_per_class) else 'N/A'}")
        print(f"    - Recall: {results.box.mr_per_class[i] if i < len(results.box.mr_per_class) else 'N/A'}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    evaluate_model()
