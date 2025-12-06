#!/usr/bin/env python3
import cv2, sys, json, time, argparse
import mediapipe as mp
from pathlib import Path

# -------- Paths --------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "trained_lbph_face_recognizer_model.yml"
LABELMAP_PATH = MODELS_DIR / "label_map.json"

# -------- CLI --------
ap = argparse.ArgumentParser(description="LBPH Face Recognition using MediaPipe.")
ap.add_argument("--threshold", type=float, default=80.0,
                help="Distance cutoff: <= threshold → match; > threshold → Unknown.")
ap.add_argument("--min-conf", type=float, default=10.0,
                help="Minimum confidence percentage to accept.")
ap.add_argument("--camera", type=int, default=0,
                help="Camera index.")
ap.add_argument("--image", type=str,
                help="If set, run recognition on a single image.")
args = ap.parse_args()

# -------- Sanity checks --------
if not MODEL_PATH.exists():
    sys.exit(f"[Error] Trained LBPH model not found: {MODEL_PATH}")

if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
    sys.exit("[Error] OpenCV contrib missing. Install: pip install opencv-contrib-python")

# -------- Load recognizer --------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

# -------- Load label map --------
id_to_name = {}
if LABELMAP_PATH.exists():
    with open(LABELMAP_PATH, "r") as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
else:
    print("[Warn] No label_map.json found; raw IDs will be shown.")

# -------- MediaPipe FaceDetector --------
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# -------- Drawing styles --------
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (255, 255, 255)
FONT_THICK = 2
TAG_BG = (0, 160, 0)
TAG_H = 50
RECT_COLOR = (0, 255, 0)
RECT_THICK = 2


# ================================================================
# ---- FACE RECOGNITION FUNCTION (MediaPipe version) -------------
# ================================================================
def recognize_in_frame(frame, threshold, min_conf):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    accepted_count = 0

    if results.detections:
        for detection in results.detections:

            # Get relative bounding box
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            # --- Ensure ROI is inside image ---
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(iw, x + w)
            y2 = min(ih, y + h)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_LINEAR)
            cv2.equalizeHist(gray, gray)

            # ---- LBPH Prediction ----
            pred_id, dist = recognizer.predict(gray)
            conf_pct = max(0.0, min(100.0, 100.0 - (dist / threshold) * 100.0))

            # ---- Accept or reject ----
            if dist <= threshold and conf_pct >= min_conf:
                accepted_count += 1
                name = id_to_name.get(pred_id, f"ID:{pred_id}")

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), RECT_COLOR, RECT_THICK)

                # Label box above face
                top = max(0, y1 - TAG_H - 10)
                cv2.rectangle(frame, (x1, top), (x2, top + TAG_H), TAG_BG, -1)

                label = f"{name}: {conf_pct:.1f}% (d={dist:.1f})"
                cv2.putText(frame, label, (x1 + 4, top + TAG_H - 15),
                            FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)

    return frame, accepted_count

# -------- Single Image Mode -------------------------------------
if args.image:
    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"[Error] Image not found: {img_path}")

    frame = cv2.imread(str(img_path))
    if frame is None:
        sys.exit(f"[Error] Could not read image: {img_path}")

    out, count = recognize_in_frame(frame, args.threshold, args.min_conf)
    cv2.putText(out, f"Accepted: {count}", (10, 25), FONT, 0.6, (0, 255, 0), 2)

    cv2.imshow("Recognition Result", out)
    print(f"[info] {img_path.name}: accepted = {count}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(0)

# -------- Webcam Mode -------------------------------------------

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    sys.exit("[Error] Could not open camera.")

prev = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("[Warn] Failed to read frame.")
        break

    out, count = recognize_in_frame(frame, args.threshold, args.min_conf)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (now - prev)))
    prev = now

    hud = f"Accepted: {count}  FPS: {fps:.1f}  thr={args.threshold}  minConf={args.min_conf}%"
    cv2.putText(out, hud, (10, 25), FONT, 0.6, (0, 255, 0), 2)

    cv2.imshow("LBPH Recognition (MediaPipe)", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()