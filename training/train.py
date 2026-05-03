from ultralytics import YOLO
import torch

def main():
    print("CUDA:", torch.cuda.is_available())
    
    # Gunakan model yang lebih akurat (medium/large)
    # yolov8n.pt = nano (cepat, kurang akurat)
    # yolov8m.pt = medium (balance)
    # yolov8l.pt = large (akurat, lambat)
    model = YOLO("yolov8m.pt")  # ⬆️ UPGRADED ke Medium untuk akurasi lebih baik

    model.train(
        data="training/data.yaml",
        epochs=100,  # ⬆️ NAIKKAN epochs untuk convergence lebih baik
        imgsz=640,   # ⬆️ NAIKKAN ke 640 untuk detail lebih baik (standar YOLO)
        
        batch=8,     # ⬆️ NAIKKAN batch size jika GPU cukup
        workers=0,   # ✓ WAJIB di Windows
        device=0,    # GPU 0
        
        name="road_damage_model_v2",
        
        # === DATA AUGMENTATION STRATEGY ===
        hsv_h=0.015,  # Hue variation
        hsv_s=0.75,   # ⬆️ LEBIH STRONG saturation
        hsv_v=0.4,    # ⬆️ Value variation
        
        degrees=15.0,     # ⬆️ LEBIH BANYAK rotation
        translate=0.1,    # ⬆️ LEBIH BANYAK translation
        scale=0.5,        # ⬆️ LEBIH BANYAK scale variation
        
        fliplr=0.5,   # Horizontal flip
        flipud=0.1,   # ⬆️ TAMBAHKAN vertical flip
        
        # === REGULARIZATION ===
        perspective=0.0,  # 3D perspective
        mixup=0.0,        # Mixup augmentation
        mosaic=1.0,       # Mosaic augmentation (SANGAT MEMBANTU)
        
        # === TRAINING PARAMETERS ===
        optimizer="SGD",  # ✓ SGD lebih stabil untuk YOLO
        lr0=0.01,         # Learning rate
        lrf=0.01,         # Final LR ratio
        momentum=0.937,   # Momentum untuk SGD
        weight_decay=0.0005,  # L2 regularization
        
        patience=20,      # Early stopping
        save=True,
        save_period=10,   # Save checkpoint setiap 10 epochs
        
        # === VALIDATION ===
        val=True,
        plots=True,
        
        # === PRECISION & RECALL ===
        conf=0.5,         # Confidence threshold untuk training
    )

if __name__ == "__main__":
    main()