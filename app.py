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

SEVERITY_THRESHOLDS = {
    # (area_pct_threshold, conf_threshold) → severity
    "berat":  {"area_pct": 3.0,  "conf": 0.60},
    "sedang": {"area_pct": 0.8,  "conf": 0.40},
    "ringan": {"area_pct": 0.0,  "conf": 0.0},
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


def compute_severity(conf: float, area_pct: float) -> dict:
    """Return severity label + score (0-100) based on confidence and area."""
    if conf >= SEVERITY_THRESHOLDS["berat"]["conf"] or area_pct >= SEVERITY_THRESHOLDS["berat"]["area_pct"]:
        label = "Berat"
        score = min(100, int(50 + conf * 30 + area_pct * 5))
        color = "#ef4444"
        icon  = "🔴"
    elif conf >= SEVERITY_THRESHOLDS["sedang"]["conf"] or area_pct >= SEVERITY_THRESHOLDS["sedang"]["area_pct"]:
        label = "Sedang"
        score = min(79, int(30 + conf * 30 + area_pct * 5))
        color = "#f59e0b"
        icon  = "🟡"
    else:
        label = "Ringan"
        score = max(10, int(conf * 45 + area_pct * 3))
        color = "#22c55e"
        icon  = "🟢"
    return {"label": label, "score": score, "color": color, "icon": icon}


def hex_to_bgr(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def draw_filled_box(img, x1, y1, x2, y2, bgr_color, alpha=0.18):
    """Draw a semi-transparent filled rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def adaptive_font_scale(img_w, img_h):
    """Return font scale adaptive to image resolution."""
    base = min(img_w, img_h)
    scale = max(0.45, min(1.2, base / 800))
    thickness = max(1, int(scale * 2))
    return scale, thickness


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

        # ── Run inference (optimized params) ─────────────────────────────────
        results = model.predict(
            source=str(upload_path),
            conf=0.15,          # Lower threshold → detect faint/light damage
            iou=0.35,           # Tighter NMS → keep more distinct detections
            imgsz=1280,         # Higher resolution → more detail captured
            augment=True,       # Test-Time Augmentation (TTA) for better recall
            agnostic_nms=True,  # Class-agnostic NMS → better for overlapping
            save=False,
            verbose=False,
        )

        result = results[0]

        # ── Draw annotated image ─────────────────────────────────────────────
        img_bgr = cv2.imread(str(upload_path))
        h, w = img_bgr.shape[:2]
        total_img_area = w * h
        font_scale, font_thickness = adaptive_font_scale(w, h)
        box_thickness = max(2, int(font_scale * 3))

        detections = []
        summary    = {}
        severity_counts = {"Berat": 0, "Sedang": 0, "Ringan": 0}

        boxes = result.boxes
        if boxes is not None and len(boxes):
            # Sort by confidence desc so high-conf boxes drawn last (on top)
            box_list = sorted(boxes, key=lambda b: float(b.conf[0]))

            for box in box_list:
                cls_id   = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                meta      = CLASS_META.get(cls_name, {"label": cls_name, "color": "#ffffff", "emoji": "🔍"})
                bgr_color = hex_to_bgr(meta["color"])

                # Area calculations
                box_area  = (x2 - x1) * (y2 - y1)
                area_pct  = round(box_area / total_img_area * 100, 2)

                # Severity
                sev = compute_severity(conf, area_pct)
                severity_counts[sev["label"]] += 1

                # ── Draw: semi-transparent fill ───────────────────────────────
                draw_filled_box(img_bgr, x1, y1, x2, y2, bgr_color, alpha=0.20)

                # ── Draw: bounding box outline ────────────────────────────────
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), bgr_color, box_thickness)

                # ── Draw: corner accents ──────────────────────────────────────
                corner_len = max(12, int(min(x2 - x1, y2 - y1) * 0.15))
                cth = box_thickness + 1
                for cx, cy, dx, dy in [
                    (x1, y1, 1, 1), (x2, y1, -1, 1),
                    (x1, y2, 1, -1), (x2, y2, -1, -1),
                ]:
                    cv2.line(img_bgr, (cx, cy), (cx + dx * corner_len, cy), bgr_color, cth)
                    cv2.line(img_bgr, (cx, cy), (cx, cy + dy * corner_len), bgr_color, cth)

                # ── Draw: label tag ───────────────────────────────────────────
                label_main = f"{meta['label']} {conf:.0%}"
                label_sub  = f"{sev['icon']} {sev['label']} | Area: {area_pct:.1f}%"

                (tw1, th1), _ = cv2.getTextSize(label_main, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                (tw2, th2), _ = cv2.getTextSize(label_sub,  cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, 1)

                tag_w   = max(tw1, tw2) + 14
                tag_h   = th1 + th2 + 18
                tag_y1  = max(0, y1 - tag_h)
                tag_y2  = y1

                # Tag background (solid fill)
                cv2.rectangle(img_bgr, (x1, tag_y1), (x1 + tag_w, tag_y2), bgr_color, -1)

                # Main label text
                cv2.putText(
                    img_bgr, label_main,
                    (x1 + 6, tag_y2 - th2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_thickness, cv2.LINE_AA,
                )
                # Sub label text (severity + area)
                cv2.putText(
                    img_bgr, label_sub,
                    (x1 + 6, tag_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75,
                    (220, 220, 220), 1, cv2.LINE_AA,
                )

                detections.append({
                    "class":       cls_name,
                    "label":       meta["label"],
                    "color":       meta["color"],
                    "emoji":       meta["emoji"],
                    "confidence":  round(conf, 4),
                    "bbox":        [x1, y1, x2, y2],
                    "area_px":     box_area,
                    "area_pct":    area_pct,
                    "severity":    sev["label"],
                    "severity_score": sev["score"],
                    "severity_color": sev["color"],
                    "severity_icon":  sev["icon"],
                    "width":       w,
                    "height":      h,
                })
                summary[cls_name] = summary.get(cls_name, 0) + 1

        # ── Watermark ─────────────────────────────────────────────────────────
        wm_text = f"Road Damage AI | {len(detections)} deteksi"
        (wm_w, wm_h), _ = cv2.getTextSize(wm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        wm_x, wm_y = 10, h - 12
        cv2.rectangle(img_bgr, (wm_x - 4, wm_y - wm_h - 6), (wm_x + wm_w + 4, wm_y + 4),
                      (0, 0, 0), -1)
        cv2.putText(img_bgr, wm_text, (wm_x, wm_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # Save result image (high quality)
        cv2.imwrite(str(result_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

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

        # Overall severity assessment
        if severity_counts["Berat"] > 0:
            overall_severity = "Berat"
            overall_color    = "#ef4444"
            overall_icon     = "🔴"
            overall_desc     = "Jalan memerlukan perbaikan segera!"
        elif severity_counts["Sedang"] > 0:
            overall_severity = "Sedang"
            overall_color    = "#f59e0b"
            overall_icon     = "🟡"
            overall_desc     = "Jalan memerlukan perhatian dan perbaikan."
        elif severity_counts["Ringan"] > 0:
            overall_severity = "Ringan"
            overall_color    = "#22c55e"
            overall_icon     = "🟢"
            overall_desc     = "Kerusakan minor, pantau secara berkala."
        else:
            overall_severity = "Baik"
            overall_color    = "#3b82f6"
            overall_icon     = "✅"
            overall_desc     = "Tidak ada kerusakan terdeteksi."

        total_area_pct = round(sum(d["area_pct"] for d in detections), 2)

        return jsonify({
            "success":            True,
            "result_image_url":   f"/static/results/{result_path.name}",
            "detections":         detections,
            "summary":            summary_list,
            "total":              len(detections),
            "severity_counts":    severity_counts,
            "overall_severity":   overall_severity,
            "overall_color":      overall_color,
            "overall_icon":       overall_icon,
            "overall_desc":       overall_desc,
            "total_damage_area_pct": total_area_pct,
            "image_width":        w,
            "image_height":       h,
        })

    except Exception as e:
        return jsonify({"error": f"Prediksi gagal: {str(e)}"}), 500


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
