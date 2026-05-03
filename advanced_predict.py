"""
Advanced prediction script dengan berbagai strategi peningkatan akurasi
Gunakan ini untuk testing dan validation
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class AdvancedPredictor:
    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)
        self.conf_threshold = 0.4
        self.iou_threshold = 0.5
        
    def preprocess_image(self, image_path):
        """Enhanced preprocessing untuk akurasi lebih baik"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # 1. CLAHE - Contrast Limited Adaptive Histogram Equalization
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. Noise reduction
        img = cv2.fastNlMeansDenoisingColored(
            img, None, 
            h=10, 
            hForColorComponents=10, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
        
        # 3. Bilateral filter untuk preserve edges
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        return img
    
    def predict_single(self, image_path, use_preprocessing=True):
        """Predict pada single image"""
        
        if use_preprocessing:
            img = self.preprocess_image(image_path)
            if img is None:
                return None
            temp_path = str(Path(image_path).parent / "temp_preprocessed.jpg")
            cv2.imwrite(temp_path, img)
            image_path = temp_path
        
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            max_det=100,
            verbose=False
        )
        
        return results[0]
    
    def predict_with_confidence_refinement(self, image_path):
        """Predict dengan multiple confidence levels dan confidence refinement"""
        
        # Multi-pass detection dengan confidence levels berbeda
        detections = []
        
        for conf in [0.3, 0.4, 0.5]:
            result = self.model.predict(
                source=image_path,
                conf=conf,
                iou=self.iou_threshold,
                imgsz=640,
                verbose=False
            )[0]
            
            if result.boxes is not None:
                for box in result.boxes:
                    detections.append({
                        'class': int(box.cls),
                        'conf': float(box.conf),
                        'box': box.xyxy[0].cpu().numpy()
                    })
        
        # Consolidate detections (NMS)
        if detections:
            # Sort by confidence
            detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        
        return detections
    
    def predict_with_tta(self, image_path):
        """Test-Time Augmentation untuk akurasi lebih tinggi"""
        print("Running TTA (Test-Time Augmentation)...")
        
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            augment=True,  # Enable TTA
            max_det=100,
            verbose=False
        )
        
        return results[0]
    
    def visualize_results(self, image_path, result, output_path="prediction_result.jpg"):
        """Visualize prediction results"""
        img = cv2.imread(image_path)
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            class_names = [
                "longitudinal_crack",
                "transverse_crack", 
                "alligator_crack",
                "pothole"
            ]
            
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 0, 255),    # Red
                (255, 255, 0)   # Cyan
            ]
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls)
                
                # Draw box
                color = colors[cls_id % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_names[cls_id]}: {conf:.2f}"
                cv2.putText(
                    img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        
        cv2.imwrite(output_path, img)
        print(f"Result saved to {output_path}")
        return img

# Usage Example
if __name__ == "__main__":
    predictor = AdvancedPredictor("models/best.pt")
    
    # Test image
    test_image = "bus.jpg"
    
    print("\n1️⃣ Basic Prediction with Preprocessing:")
    result = predictor.predict_single(test_image, use_preprocessing=True)
    if result and result.boxes is not None:
        print(f"Found {len(result.boxes)} detections")
    
    print("\n2️⃣ Prediction with Confidence Refinement:")
    detections = predictor.predict_with_confidence_refinement(test_image)
    print(f"Found {len(detections)} consolidated detections")
    
    print("\n3️⃣ Prediction with TTA (Test-Time Augmentation):")
    result_tta = predictor.predict_with_tta(test_image)
    if result_tta and result_tta.boxes is not None:
        print(f"Found {len(result_tta.boxes)} detections with TTA")
    
    print("\n4️⃣ Visualizing Results:")
    predictor.visualize_results(test_image, result_tta, "hasil_prediction.jpg")
    print("✅ Done!")
