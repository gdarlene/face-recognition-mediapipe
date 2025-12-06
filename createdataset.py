#!/usr/bin/env python3
import cv2
import time
import os
import mediapipe as mp

# ---- Config ----
MAX_IMAGES = 200
INTERVAL_MS = 400
FRAME_SIZE = (800, 800)  # (w, h)
DATASET_DIR = 'dataset'

# ---- Setup ----
os.makedirs(DATASET_DIR, exist_ok=True)

ID = input('Enter your ID: ').strip()
print("Please get your face ready!")
time.sleep(2)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
if not cam.isOpened():
    raise SystemExit("[Error] Could not open camera.")

win = "Dataset Generating..."
cv2.namedWindow(win)

start_time = time.time()
last_capture_ts = start_time
image_count = 0

# visual feedback flag
last_saved_flash_ts = 0.0
FLASH_MS = 200  # how long to show the 'Saved!' flash

# ---- Mediapipe face detection ----
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

while True:
    ok, frame = cam.read()
    if not ok:
        print("[Warn] Failed to read frame.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    faces = []
    if results.detections:
        for detection in results.detections:
            # get relative bounding box
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            # store
            faces.append((x, y, w, h))

    # ---- Process faces + capture ----
    for (x, y, w, h) in faces:

        # draw detection box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 0), 2)

        elapsed_ms = (time.time() - last_capture_ts) * 1000.0
        if elapsed_ms >= INTERVAL_MS and image_count < MAX_IMAGES:

            # crop safely
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            face_crop_color = frame[y1:y2, x1:x2]
            face_crop_gray = cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2GRAY)

            filename = os.path.join(
                DATASET_DIR, f"data.{ID}.{int(time.time() * 1000)}.jpg"
            )
            cv2.imwrite(filename, face_crop_gray)
            image_count += 1
            last_capture_ts = time.time()
            last_saved_flash_ts = last_capture_ts  # trigger flash

            print(f"Saved [{image_count}/{MAX_IMAGES}]: {os.path.basename(filename)}")

            break  # only 1 capture per interval

    # ---- HUD overlay ----
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    hud = f"ID: {ID}  |  Saved: {image_count}/{MAX_IMAGES}  |  Interval: {INTERVAL_MS} ms  |  Press 'q' to quit"
    cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Flash “[OK]” after saving
    if (time.time() - last_saved_flash_ts) * 1000.0 <= FLASH_MS:
        text = "[OK]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

        pad_x, pad_y = 10, 10
        x0, y0 = 20, 80

        cv2.rectangle(frame,
                      (x0 - pad_x, y0 - text_h - pad_y),
                      (x0 + text_w + pad_x, y0 + baseline + pad_y),
                      (0, 180, 0), -1)

        cv2.putText(frame, text, (x0, y0), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Progress bar
    bar_w = frame.shape[1] - 20
    pct = 0 if MAX_IMAGES == 0 else min(1.0, image_count / float(MAX_IMAGES))
    filled = int(bar_w * pct)
    y1 = frame.shape[0] - 20

    cv2.rectangle(frame, (10, y1), (10 + bar_w, y1 + 10), (255, 255, 255), 1)
    cv2.rectangle(frame, (10, y1), (10 + filled, y1 + 10), (0, 180, 0), -1)

    cv2.imshow(win, frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q') or image_count >= MAX_IMAGES:
        break

cam.release()
cv2.destroyAllWindows()

print(f"\n√ Dataset generation complete.")
print(f"   ID: {ID}")
print(f"   Saved: {image_count} image(s) to '{DATASET_DIR}/'")