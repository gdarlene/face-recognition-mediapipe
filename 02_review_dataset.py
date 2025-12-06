#!/usr/bin/env python3
"""
Dataset Preview and Cleaner (MediaPipe version)

This tool helps you browse, inspect, and clean a dataset of face images.
It supports:
- Viewing images in a fixed 1280×720 preview window
- MediaPipe-based face detection overlay during preview
- Navigating images (next/prev)
- Automatic slideshow
- Permanent deletion of bad images (no trash, no undo)
- Keyboard shortcuts explained in the HUD

Controls
--------
n          : next image
p          : previous image
space / s  : play/pause slideshow
+ or =     : faster slideshow
- or _     : slower slideshow
d          : delete current image (PERMANENT)
q or ESC   : quit
"""

import cv2
import time
import numpy as np
from pathlib import Path
from shutil import rmtree
import mediapipe as mp


# ------------------------------------------------------------------------------------
# Auto-clean old trash folders from earlier tool versions
# ------------------------------------------------------------------------------------
for trash_dir in Path("dataset").rglob(".trash"):
    try:
        rmtree(trash_dir, ignore_errors=True)
        print(f"[cleanup] Removed old trash folder: {trash_dir}")
    except Exception as e:
        print(f"[warn] Could not remove {trash_dir}: {e}")


# ------------------------------------------------------------------------------------
# MediaPipe Face Detection
# ------------------------------------------------------------------------------------
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Prepare a persistent detector
FACE_DETECTOR = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


def detect_faces_mediapipe(image):
    """
    Detect faces in a BGR image and return bounding boxes.
    Returns a list of tuples: (x, y, w, h)
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = FACE_DETECTOR.process(rgb)

    h, w = image.shape[:2]
    boxes = []

    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            w_box = int(box.width * w)
            h_box = int(box.height * h)
            boxes.append((x, y, w_box, h_box))

    return boxes


# ------------------------------------------------------------------------------------
# Preview Tool
# ------------------------------------------------------------------------------------
def preview(folder="dataset", pattern="*.jpg", delay_ms=1000):

    folder = Path(folder)

    # Gather images recursively
    files = sorted(
        f for f in folder.rglob(pattern)
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )

    if not files:
        print(f"No images found in '{folder}'")
        return

    win = "Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # UI layout config
    FRAME_W, FRAME_H = 1280, 720
    TOP_BAR_H, BOTTOM_BAR_H = 55, 55
    VIEW_X, VIEW_Y = 0, TOP_BAR_H
    VIEW_W, VIEW_H = FRAME_W, FRAME_H - TOP_BAR_H - BOTTOM_BAR_H

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (255, 255, 255)
    GREEN = (0, 180, 0)
    PAD_X, PAD_Y = 20, 15

    # Flash notification
    FLASH_TS = 0.0
    FLASH_TEXT = ""
    FLASH_MS = 400

    def truncate_to_width(text, max_w, scale, thick):
        """Truncate long filenames (add … in the middle)."""
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
        if tw <= max_w:
            return text

        if len(text) < 6:
            return text[:3] + "…"

        for cut in range(2, len(text) - 2):
            candidate = text[:cut] + "…" + text[-cut:]
            (cw, _), _ = cv2.getTextSize(candidate, FONT, scale, thick)
            if cw <= max_w:
                return candidate
        return text

    def delete_current(path: Path):
        """Delete image permanently."""
        nonlocal FLASH_TS, FLASH_TEXT
        try:
            path.unlink(missing_ok=True)
            FLASH_TEXT = "Deleted permanently"
            FLASH_TS = time.time()
            print(f"[deleted] {path}")
            return True
        except Exception as e:
            FLASH_TEXT = "Delete failed"
            FLASH_TS = time.time()
            print(f"[error] {e}")
            return False

    def draw_flash(frame):
        if (time.time() - FLASH_TS) * 1000 <= FLASH_MS and FLASH_TEXT:
            text = FLASH_TEXT
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)

            x1 = FRAME_W - PAD_X - tw - 24
            y1 = 8
            x2 = FRAME_W - PAD_X
            y2 = 8 + th + 18

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 160, 0), -1)
            cv2.putText(frame, text, (x1 + 12, y2 - 8),
                        FONT, 0.65, WHITE, 2, cv2.LINE_AA)

    def draw_frame(img_path, autoplay, delay, idx):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        # Load image
        img = cv2.imread(str(img_path))

        if img is not None and img.size > 0:

            # ---- Face Detection (MediaPipe) ----
            boxes = detect_faces_mediapipe(img)
            for (x, y, w_box, h_box) in boxes:
                cv2.rectangle(img, (x, y), (x + w_box, y + h_box),
                              (0, 255, 0), 2)

            # Letterboxing for preview
            ih, iw = img.shape[:2]
            scale = min(VIEW_W / iw, VIEW_H / ih)
            new_w, new_h = int(iw * scale), int(ih * scale)
            resized = cv2.resize(img, (new_w, new_h))

            off_x = VIEW_X + (VIEW_W - new_w) // 2
            off_y = VIEW_Y + (VIEW_H - new_h) // 2
            frame[off_y:off_y+new_h, off_x:off_x+new_w] = resized

        # ---- Top bar ----
        cv2.rectangle(frame, (0, 0), (FRAME_W, TOP_BAR_H), GREEN, -1)
        left = f"[{idx+1}/{len(files)}] "
        state = f"  |  {'PLAY' if autoplay else 'PAUSE'}  |  delay={delay}ms"

        max_name_w = FRAME_W - 2*PAD_X - \
            cv2.getTextSize(left + state, FONT, 0.8, 2)[0][0]

        name = truncate_to_width(img_path.name, max_name_w, 0.8, 2)
        top_text = left + name + state

        cv2.putText(frame, top_text,
                    (PAD_X, TOP_BAR_H - PAD_Y),
                    FONT, 0.8, WHITE, 2)

        # ---- Bottom bar ----
        cv2.rectangle(frame,
                      (0, FRAME_H - BOTTOM_BAR_H),
                      (FRAME_W, FRAME_H),
                      GREEN, -1)

        help_text = (
            "p: prev   n: next   space/s: pause/resume   "
            "+/- speed   d: DELETE (PERMANENT)   q/ESC: quit"
        )

        cv2.putText(frame, help_text,
                    (PAD_X, FRAME_H - PAD_Y),
                    FONT, 0.65, WHITE, 2)

        # Flash notification
        draw_flash(frame)

        return frame

    # ------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------
    idx = 0
    autoplay = False
    last = time.time()

    KEY_LEFT = 2424832
    KEY_RIGHT = 2555904

    while True:

        if not files:
            blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            cv2.rectangle(blank, (0, 0), (FRAME_W, TOP_BAR_H), GREEN, -1)
            cv2.putText(blank, "No images left. Press q to exit.",
                        (PAD_X, TOP_BAR_H - PAD_Y),
                        FONT, 0.8, WHITE, 2)
            cv2.imshow(win, blank)
        else:
            frame = draw_frame(files[idx], autoplay, delay_ms, idx)
            cv2.imshow(win, frame)

        # Slideshow auto-advance
        if files and autoplay and (time.time() - last) * 1000 >= delay_ms:
            idx = (idx + 1) % len(files)
            last = time.time()

        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        # Quit
        if key in (ord('q'), ord('Q'), 27):
            break

        # Next / Prev
        if files and key in (KEY_RIGHT, ord('n'), ord('N')):
            idx = (idx + 1) % len(files)
            last = time.time()
        elif files and key in (KEY_LEFT, ord('p'), ord('P')):
            idx = (idx - 1) % len(files)
            last = time.time()

        # Play/Pause
        elif key in (ord(' '), ord('s'), ord('S')):
            autoplay = not autoplay
            last = time.time()

        # Speed control
        elif key in (ord('+'), ord('=')):
            delay_ms = max(50, int(delay_ms * 0.8))
        elif key in (ord('-'), ord('_')):
            delay_ms = min(5000, int(delay_ms * 1.25))

        # Delete permanently
        elif files and key in (ord('d'), ord('D')):
            to_delete = files[idx]
            if delete_current(to_delete):
                files.pop(idx)
                if files:
                    idx %= len(files)

    cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    preview(folder="dataset", pattern="*.jpg", delay_ms=250)