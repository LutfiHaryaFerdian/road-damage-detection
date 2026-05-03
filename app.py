import os
import uuid
import time
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "models" / "best.pt"
UPLOAD_DIR  = BASE_DIR / "static" / "uploads"
RESULT_DIR  = BASE_DIR / "static" / "results"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024          # 16 MB max upload
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Class metadata ───────────────────────────────────────────────────────────
CLASS_META = {
    "longitudinal_crack": {"label": "Retak Memanjang",   "color": "#f59e0b", "emoji": "↔️"},
    "transverse_crack":   {"label": "Retak Melintang",   "color": "#ef4444", "emoji": "↕️"},
    "alligator_crack":    {"label": "Retak Kulit Buaya", "color": "#8b5cf6", "emoji": "🐊"},
    "pothole":            {"label": "Lubang",             "color": "#06b6d4", "emoji": "⚠️"},
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Load model once at startup ────────────────────────────────────────────────
print(f"[Road Damage AI] Loading model from {MODEL_PATH} ...")
model = YOLO(str(MODEL_PATH))
print("[Road Damage AI] Model loaded successfully.")


# ── Helpers ──────────────────────────────────────────────────────────────────
def cleanup_old_files(directory: Path, max_age_seconds: int = 3600):
    """Remove files older than max_age_seconds to save disk space."""
    now = time.time()
    for f in directory.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > max_age_seconds:
            f.unlink(missing_ok=True)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    cleanup_old_files(UPLOAD_DIR)
    cleanup_old_files(RESULT_DIR)

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung. Gunakan JPG, PNG, atau WEBP."}), 400

    try:
        ext = Path(file.filename).suffix.lower()
        uid = uuid.uuid4().hex
        upload_path = UPLOAD_DIR / f"{uid}{ext}"
        result_path = RESULT_DIR / f"{uid}_result.jpg"

        # Save upload
        file.save(str(upload_path))

        # Run inference
        results = model.predict(
            source=str(upload_path),
            conf=0.25,
            iou=0.45,
            save=False,
            verbose=False,
        )

        result = results[0]

        # ── Draw annotated image manually (full control over colors) ────────
        img_bgr = cv2.imread(str(upload_path))
        h, w = img_bgr.shape[:2]

        detections = []
        summary = {}  # class_name → count

        boxes = result.boxes
        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id    = int(box.cls[0])
                cls_name  = model.names[cls_id]
                conf      = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                meta  = CLASS_META.get(cls_name, {"label": cls_name, "color": "#ffffff", "emoji": "🔍"})
                color_hex = meta["color"].lstrip("#")
                r, g, b   = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
                bgr_color  = (b, g, r)

                # Draw box
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), bgr_color, 2)

                # Label background
                label_text = f"{meta['label']} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 8, y1), bgr_color, -1)
                cv2.putText(
                    img_bgr, label_text,
                    (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
                )

                detections.append({
                    "class":      cls_name,
                    "label":      meta["label"],
                    "color":      meta["color"],
                    "emoji":      meta["emoji"],
                    "confidence": round(conf, 4),
                    "bbox":       [x1, y1, x2, y2],
                    "width":      w,
                    "height":     h,
                })
                summary[cls_name] = summary.get(cls_name, 0) + 1

        # Save result image
        cv2.imwrite(str(result_path), img_bgr)

        # Build summary list
        summary_list = [
            {
                "class": k,
                "label": CLASS_META.get(k, {}).get("label", k),
                "color": CLASS_META.get(k, {}).get("color", "#fff"),
                "emoji": CLASS_META.get(k, {}).get("emoji", "🔍"),
                "count": v,
            }
            for k, v in summary.items()
        ]

        return jsonify({
            "success":          True,
            "result_image_url": f"/static/results/{result_path.name}",
            "detections":       detections,
            "summary":          summary_list,
            "total":            len(detections),
        })

    except Exception as e:
        return jsonify({"error": f"Prediksi gagal: {str(e)}"}), 500


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
