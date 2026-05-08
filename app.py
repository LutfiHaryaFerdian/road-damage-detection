import os
import uuid
import time
import json
import base64
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
VIDEO_DIR  = BASE_DIR / "static" / "videos"

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_META = {
    "longitudinal_crack": {"label": "Retak Memanjang",   "color": "#f59e0b", "emoji": "↔️"},
    "transverse_crack":   {"label": "Retak Melintang",   "color": "#ef4444", "emoji": "↕️"},
    "alligator_crack":    {"label": "Retak Kulit Buaya", "color": "#8b5cf6", "emoji": "🐊"},
    "pothole":            {"label": "Lubang",             "color": "#06b6d4", "emoji": "⚠️"},
}

SEVERITY_THRESHOLDS = {
    "berat":  {"area_pct": 3.0, "conf": 0.60},
    "sedang": {"area_pct": 0.8, "conf": 0.40},
    "ringan": {"area_pct": 0.0, "conf": 0.0},
}

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}

# ── Video progress tracking ───────────────────────────────────────────────────
video_progress: dict = {}  # uid → dict

# ── Load model ────────────────────────────────────────────────────────────────
START_TIME = time.time()
print(f"[Road Damage AI] Loading model from {MODEL_PATH} ...")
model = YOLO(str(MODEL_PATH))
print("[Road Damage AI] Model loaded OK.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def cleanup_old_files(directory: Path, max_age_seconds: int = 3600):
    now = time.time()
    for f in directory.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > max_age_seconds:
            f.unlink(missing_ok=True)


def compute_severity(conf: float, area_pct: float) -> dict:
    if conf >= SEVERITY_THRESHOLDS["berat"]["conf"] or area_pct >= SEVERITY_THRESHOLDS["berat"]["area_pct"]:
        return {"label": "Berat",  "score": min(100, int(50 + conf*30 + area_pct*5)), "color": "#ef4444", "icon": "🔴"}
    elif conf >= SEVERITY_THRESHOLDS["sedang"]["conf"] or area_pct >= SEVERITY_THRESHOLDS["sedang"]["area_pct"]:
        return {"label": "Sedang", "score": min(79,  int(30 + conf*30 + area_pct*5)), "color": "#f59e0b", "icon": "🟡"}
    else:
        return {"label": "Ringan", "score": max(10,  int(conf*45 + area_pct*3)),      "color": "#22c55e", "icon": "🟢"}


def hex_to_bgr(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def draw_filled_box(img, x1, y1, x2, y2, bgr_color, alpha=0.18):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def adaptive_font_scale(img_w, img_h):
    base  = min(img_w, img_h)
    scale = max(0.45, min(1.2, base / 800))
    return scale, max(1, int(scale * 2))


def draw_detections_on_frame(frame, result, model_names):
    """Draw bboxes on a frame in-place. Returns detection count."""
    h, w = frame.shape[:2]
    font_scale, font_thickness = adaptive_font_scale(w, h)
    box_thickness = max(2, int(font_scale * 3))
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0
    total = 0
    for box in sorted(boxes, key=lambda b: float(b.conf[0])):
        cls_id   = int(box.cls[0])
        cls_name = model_names[cls_id]
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        meta      = CLASS_META.get(cls_name, {"label": cls_name, "color": "#ffffff"})
        bgr_color = hex_to_bgr(meta["color"])

        draw_filled_box(frame, x1, y1, x2, y2, bgr_color, 0.20)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, box_thickness)

        corner_len = max(12, int(min(x2-x1, y2-y1) * 0.15))
        cth = box_thickness + 1
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + dx*corner_len, cy), bgr_color, cth)
            cv2.line(frame, (cx, cy), (cx, cy + dy*corner_len), bgr_color, cth)

        label = f"{meta['label']} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        tag_y1 = max(0, y1 - th - 10)
        cv2.rectangle(frame, (x1, tag_y1), (x1+tw+10, y1), bgr_color, -1)
        cv2.putText(frame, label, (x1+5, y1-4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)
        total += 1
    return total


# ── Video worker ──────────────────────────────────────────────────────────────
def process_video_thread(uid: str, input_path: Path, output_path: Path):
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            video_progress[uid] = {"percent": 0, "status": "error", "message": "Tidak bisa membuka video"}
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(str(output_path), fourcc, fps_vid, (w, h))

        frame_idx   = 0
        total_dets  = 0
        class_counts: dict = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=0.25, iou=0.45,
                                    imgsz=640, augment=False, save=False, verbose=False)
            n = draw_detections_on_frame(frame, results[0], model.names)
            total_dets += n

            # Count classes
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    name = model.names[int(box.cls[0])]
                    class_counts[name] = class_counts.get(name, 0) + 1

            out.write(frame)
            frame_idx += 1
            pct = int(frame_idx / total_frames * 100)
            video_progress[uid] = {
                "percent": pct, "status": "processing",
                "message": f"Frame {frame_idx}/{total_frames}",
                "frame": frame_idx, "total_frames": total_frames,
            }

        cap.release()
        out.release()

        video_progress[uid] = {
            "percent": 100, "status": "done",
            "message": "Selesai!",
            "video_url": f"/static/videos/{output_path.name}",
            "total_detections": total_dets,
            "total_frames": frame_idx,
            "class_counts": class_counts,
        }
    except Exception as e:
        video_progress[uid] = {"percent": 0, "status": "error", "message": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "online",
        "model": "YOLOv8",
        "model_file": MODEL_PATH.name,
        "classes": list(model.names.values()),
        "class_meta": {k: {"label": v["label"], "color": v["color"], "emoji": v["emoji"]}
                       for k, v in CLASS_META.items()},
        "uptime_seconds": round(time.time() - START_TIME),
    })


# ── Image detection ───────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    cleanup_old_files(UPLOAD_DIR)
    cleanup_old_files(RESULT_DIR)

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400
    if Path(file.filename).suffix.lower() not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": "Format tidak didukung. Gunakan JPG/PNG/WEBP."}), 400

    try:
        ext         = Path(file.filename).suffix.lower()
        uid         = uuid.uuid4().hex
        upload_path = UPLOAD_DIR / f"{uid}{ext}"
        result_path = RESULT_DIR / f"{uid}_result.jpg"
        file.save(str(upload_path))

        results = model.predict(source=str(upload_path), conf=0.15, iou=0.35,
                                imgsz=1280, augment=True, agnostic_nms=True,
                                save=False, verbose=False)

        result    = results[0]
        img_bgr   = cv2.imread(str(upload_path))
        h, w      = img_bgr.shape[:2]
        total_area = w * h
        font_scale, font_thickness = adaptive_font_scale(w, h)
        box_thickness = max(2, int(font_scale * 3))

        detections      = []
        summary         = {}
        severity_counts = {"Berat": 0, "Sedang": 0, "Ringan": 0}

        boxes = result.boxes
        if boxes is not None and len(boxes):
            for box in sorted(boxes, key=lambda b: float(b.conf[0])):
                cls_id   = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                meta      = CLASS_META.get(cls_name, {"label": cls_name, "color": "#ffffff", "emoji": "🔍"})
                bgr_color = hex_to_bgr(meta["color"])
                box_area  = (x2-x1) * (y2-y1)
                area_pct  = round(box_area / total_area * 100, 2)
                sev       = compute_severity(conf, area_pct)
                severity_counts[sev["label"]] += 1

                draw_filled_box(img_bgr, x1, y1, x2, y2, bgr_color, 0.20)
                cv2.rectangle(img_bgr, (x1,y1), (x2,y2), bgr_color, box_thickness)

                corner_len = max(12, int(min(x2-x1, y2-y1) * 0.15))
                cth = box_thickness + 1
                for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(img_bgr, (cx, cy), (cx+dx*corner_len, cy), bgr_color, cth)
                    cv2.line(img_bgr, (cx, cy), (cx, cy+dy*corner_len), bgr_color, cth)

                lbl_main = f"{meta['label']} {conf:.0%}"
                lbl_sub  = f"{sev['icon']} {sev['label']} | Area: {area_pct:.1f}%"
                (tw1, th1), _ = cv2.getTextSize(lbl_main, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                (tw2, th2), _ = cv2.getTextSize(lbl_sub,  cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.75, 1)
                tag_w  = max(tw1, tw2) + 14
                tag_h  = th1 + th2 + 18
                tag_y1 = max(0, y1 - tag_h)
                cv2.rectangle(img_bgr, (x1, tag_y1), (x1+tag_w, y1), bgr_color, -1)
                cv2.putText(img_bgr, lbl_main, (x1+6, y1-th2-8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)
                cv2.putText(img_bgr, lbl_sub,  (x1+6, y1-4),     cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.75, (220,220,220), 1, cv2.LINE_AA)

                detections.append({
                    "class": cls_name, "label": meta["label"],
                    "color": meta["color"], "emoji": meta["emoji"],
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "area_pct": area_pct,
                    "severity": sev["label"], "severity_score": sev["score"],
                    "severity_color": sev["color"], "severity_icon": sev["icon"],
                })
                summary[cls_name] = summary.get(cls_name, 0) + 1

        # Watermark
        wm = f"Road Damage AI | {len(detections)} deteksi"
        (wm_w, wm_h), _ = cv2.getTextSize(wm, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img_bgr, (6, h-wm_h-16), (14+wm_w, h-6), (0,0,0), -1)
        cv2.putText(img_bgr, wm, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)

        cv2.imwrite(str(result_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        summary_list = [{"class": k, "label": CLASS_META.get(k,{}).get("label",k),
                         "color": CLASS_META.get(k,{}).get("color","#fff"),
                         "emoji": CLASS_META.get(k,{}).get("emoji","🔍"), "count": v}
                        for k, v in summary.items()]

        if severity_counts["Berat"] > 0:
            overall = ("Berat", "#ef4444", "🔴", "Jalan memerlukan perbaikan segera!")
        elif severity_counts["Sedang"] > 0:
            overall = ("Sedang", "#f59e0b", "🟡", "Jalan memerlukan perhatian dan perbaikan.")
        elif severity_counts["Ringan"] > 0:
            overall = ("Ringan", "#22c55e", "🟢", "Kerusakan minor, pantau secara berkala.")
        else:
            overall = ("Baik", "#3b82f6", "✅", "Tidak ada kerusakan terdeteksi.")

        return jsonify({
            "success": True,
            "result_image_url": f"/static/results/{result_path.name}",
            "upload_image_url": f"/static/uploads/{upload_path.name}",
            "detections": detections, "summary": summary_list,
            "total": len(detections), "severity_counts": severity_counts,
            "overall_severity": overall[0], "overall_color": overall[1],
            "overall_icon": overall[2], "overall_desc": overall[3],
            "total_damage_area_pct": round(sum(d["area_pct"] for d in detections), 2),
            "image_width": w, "image_height": h,
        })
    except Exception as e:
        return jsonify({"error": f"Prediksi gagal: {str(e)}"}), 500


# ── Webcam frame detection ────────────────────────────────────────────────────
@app.route("/predict/frame", methods=["POST"])
def predict_frame():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400
    try:
        img_b64  = data["image"].split(",")[-1]
        img_bytes = base64.b64decode(img_b64)
        nparr    = np.frombuffer(img_bytes, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Invalid image"}), 400

        t0      = time.time()
        results = model.predict(source=img_bgr, conf=0.25, iou=0.45,
                                imgsz=640, augment=False, save=False, verbose=False)
        inf_ms  = round((time.time() - t0) * 1000)
        fps     = round(1000 / max(1, inf_ms), 1)

        result = results[0]
        h, w   = img_bgr.shape[:2]
        detections = []

        boxes = result.boxes
        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id   = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                meta = CLASS_META.get(cls_name, {"label": cls_name, "color": "#ffffff", "emoji": "🔍"})
                detections.append({
                    "class": cls_name, "label": meta["label"],
                    "color": meta["color"], "emoji": meta["emoji"],
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                })

        return jsonify({"success": True, "detections": detections,
                        "fps": fps, "inf_ms": inf_ms,
                        "total": len(detections), "width": w, "height": h})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Video detection ───────────────────────────────────────────────────────────
@app.route("/predict/video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if Path(file.filename).suffix.lower() not in ALLOWED_VIDEO_EXT:
        return jsonify({"error": "Format video tidak didukung (mp4, mov, avi, mkv)"}), 400

    uid         = uuid.uuid4().hex
    ext         = Path(file.filename).suffix.lower()
    input_path  = UPLOAD_DIR / f"{uid}{ext}"
    output_path = VIDEO_DIR / f"{uid}_result.mp4"
    file.save(str(input_path))

    video_progress[uid] = {"percent": 0, "status": "processing", "message": "Memulai proses..."}
    t = threading.Thread(target=process_video_thread, args=(uid, input_path, output_path), daemon=True)
    t.start()

    return jsonify({"success": True, "uid": uid})


@app.route("/predict/video/progress/<uid>")
def video_progress_sse(uid):
    def generate():
        while True:
            prog = video_progress.get(uid, {"percent": 0, "status": "waiting", "message": "Menunggu..."})
            yield f"data: {json.dumps(prog)}\n\n"
            if prog.get("status") in ("done", "error"):
                break
            time.sleep(0.5)
    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
