#!/usr/bin/env python3
import cv2
import time
import os
import mediapipe as mp
import math

# ---- Config ----
MAX_IMAGES = 200
INTERVAL_MS = 400
FRAME_SIZE = (800, 800)  # (w, h)
DATASET_DIR = 'dataset_mesh'

# ---- Setup ----
os.makedirs(DATASET_DIR, exist_ok=True)

ID = input('Enter your ID: ').strip()
print("Get your face ready...")
time.sleep(2)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
if not cam.isOpened():
    raise SystemExit("[Error] Camera not available.")

win = "Mesh Dataset"
cv2.namedWindow(win)

start_time = time.time()
last_capture_ts = start_time
image_count = 0
last_saved_flash_ts = 0.0
FLASH_MS = 200

# ---- Mediapipe Face Mesh ----
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

drawing_spec = mp_draw.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1)
mesh_spec = mp_draw.DrawingSpec(color=(255, 0, 255), thickness=1)

eye_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
lips_spec = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
nose_spec = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def draw_iris_circle(img, face_landmarks, indices):
    h, w, _ = img.shape
    cx = int(face_landmarks.landmark[indices[0]].x * w)
    cy = int(face_landmarks.landmark[indices[0]].y * h)

    d = []
    for idx in indices[1:]:
        px = int(face_landmarks.landmark[idx].x * w)
        py = int(face_landmarks.landmark[idx].y * h)
        d.append(math.dist([cx, cy], [px, py]))

    if d:
        r = int(sum(d) / len(d))
        cv2.circle(img, (cx, cy), r, (0, 255, 255), 2)


def mesh_to_bbox(face_landmarks, img_w, img_h):
    xs = [lm.x * img_w for lm in face_landmarks.landmark]
    ys = [lm.y * img_h for lm in face_landmarks.landmark]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    return x1, y1, x2 - x1, y2 - y1


with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as mesh:

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if results.multi_face_landmarks:
            fl = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            # ---- Draw mesh regions ----
            mp_draw.draw_landmarks(
                frame, fl,
                mp_face_mesh.FACEMESH_TESSELATION,
                drawing_spec, mesh_spec
            )

            mp_draw.draw_landmarks(
                frame, fl,
                mp_face_mesh.FACEMESH_LEFT_EYE,
                None, eye_spec
            )
            mp_draw.draw_landmarks(
                frame, fl,
                mp_face_mesh.FACEMESH_RIGHT_EYE,
                None, eye_spec
            )

            mp_draw.draw_landmarks(
                frame, fl,
                mp_face_mesh.FACEMESH_LIPS,
                None, lips_spec
            )

            mp_draw.draw_landmarks(
                frame, fl,
                mp_face_mesh.FACEMESH_NOSE,
                None, nose_spec
            )

            # Iris circles
            draw_iris_circle(frame, fl, LEFT_IRIS)
            draw_iris_circle(frame, fl, RIGHT_IRIS)

            # ---- Build bounding box from mesh ----
            x, y, bw, bh = mesh_to_bbox(fl, w, h)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)

            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

            # ---- Save at interval ----
            if (time.time() - last_capture_ts) * 1000 >= INTERVAL_MS and image_count < MAX_IMAGES:
                crop_color = frame[y1:y2, x1:x2]
                crop_gray = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)

                filename = os.path.join(
                    DATASET_DIR, f"mesh.{ID}.{int(time.time()*1000)}.jpg"
                )
                cv2.imwrite(filename, crop_gray)

                image_count += 1
                last_capture_ts = time.time()
                last_saved_flash_ts = last_capture_ts

                print(f"Saved [{image_count}/{MAX_IMAGES}] {filename}")

        # ---- HUD ----
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        hud = f"ID: {ID} | Saved: {image_count}/{MAX_IMAGES} | Press 'q' to quit"
        cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Flash "[OK]"
        if (time.time() - last_saved_flash_ts) * 1000 <= FLASH_MS:
            cv2.putText(frame, "[OK]", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

        # Progress bar
        bar_w = frame.shape[1] - 20
        pct = min(1.0, image_count / MAX_IMAGES)
        filled = int(bar_w * pct)
        yb = frame.shape[0] - 20

        cv2.rectangle(frame, (10, yb), (10 + bar_w, yb + 10), (255, 255, 255), 1)
        cv2.rectangle(frame, (10, yb), (10 + filled, yb + 10), (0, 180, 0), -1)

        cv2.imshow(win, frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= MAX_IMAGES:
            break

cam.release()
cv2.destroyAllWindows()

print("\n√ Dataset generation complete.")
print(f"   ID: {ID}")
print(f"   Saved: {image_count} images → '{DATASET_DIR}/'")