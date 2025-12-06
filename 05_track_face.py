#!/usr/bin/env python3
import cv2
import time
import numpy as np
import sys
import serial
import mediapipe as mp

# -------------------------- CONFIG --------------------------
CAM_IDX = 0
FRAME_SIZE = (800, 600)
FONT = cv2.FONT_HERSHEY_SIMPLEX

SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

# Motor tuning
MAX_ROTATE_STEP = 10       # max degrees sent per command
THRESHOLD = 20             # min pixels before adjusting
SMOOTH_FACTOR = 0.04       # lower = smoother movement
SEND_INTERVAL = 0.05       # delay between Arduino commands
# -------------------------------------------------------------

def main():
    # ---- Connect to Arduino ----
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"[OK] Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print("[ERROR] Could not connect to Arduino:", e)
        arduino = None

    # ---- MediaPipe Face Detector ----
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # ---- Camera ----
    cap = cv2.VideoCapture(CAM_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open camera")

    frame_center_x = FRAME_SIZE[0] // 2
    last_send = 0

    print("Press 'q' to quit\n")

    # ---- Main Loop ----
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe requires RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        if result.detections:
            # pick largest face
            detections = result.detections
            biggest = None
            max_area = 0

            for det in detections:
                box = det.location_data.relative_bounding_box
                box_w = int(box.width * w)
                box_h = int(box.height * h)
                area = box_w * box_h

                if area > max_area:
                    max_area = area
                    biggest = det

            if biggest is not None:
                box = biggest.location_data.relative_bounding_box
                x = int(box.xmin * w)
                y = int(box.ymin * h)
                bw = int(box.width * w)
                bh = int(box.height * h)

                cx = x + bw // 2   # face center x-coordinate
                offset = cx - frame_center_x

                # Draw box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (cx, y + bh // 2), 5, (0, 0, 255), -1)

                # ---- Smooth servo control ----
                if abs(offset) > THRESHOLD and (time.time() - last_send > SEND_INTERVAL):
                    rotate_deg = np.clip(abs(offset) * SMOOTH_FACTOR, 1, MAX_ROTATE_STEP)

                    if offset < 0:
                        command = f"CCW {rotate_deg:.1f}\n"
                        print(f"Left ({offset}) → CCW {rotate_deg:.1f}°")
                    else:
                        command = f"CW {rotate_deg:.1f}\n"
                        print(f"Right ({offset}) → CW {rotate_deg:.1f}°")

                    if arduino:
                        arduino.write(command.encode('utf-8'))

                    last_send = time.time()

        cv2.imshow("MediaPipe Face Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---- Cleanup ----
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == "__main__":
    main()