#!/usr/bin/env python3
"""
Train LBPH recognizer using MediaPipe instead of Haar Cascade.

- Loads dataset/images
- Uses MediaPipe Face Detection to detect a face
- Crops + resizes to 200x200
- Trains LBPH (OpenCV)
- Optionally performs validation
- Saves model + label map
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# -------- Paths --------
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATASET = ROOT / "dataset"
MODEL_YML = MODELS / "trained_lbph_face_recognizer_model.yml"
LABELMAP = MODELS / "label_map.json"

# -------- CLI --------
ap = argparse.ArgumentParser(description="Train LBPH with MediaPipe face detection.")
ap.add_argument("--val-split", type=float, default=0.0)
ap.add_argument("--threshold", type=float, default=80.0)
ap.add_argument("--unknown-csv", default="")
args = ap.parse_args()

# -------- MediaPipe init --------
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# -------- Helpers --------
def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in sorted(folder.rglob("*")):
        if ".trash" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
            yield p

def label_from_name(p: Path):
    parent = p.parent.name
    if parent not in ('.trash', 'dataset'):
        return parent

    parts = p.name.split(".")
    if len(parts) > 2 and parts[0].lower() == "data":
        return parts[1]

    stem = p.stem
    if "_" in stem:
        return stem.split("_", 1)[0]

    return None

def detect_with_mediapipe(gray):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    res = detector.process(rgb)

    if not res.detections:
        return None

    h, w = gray.shape
    d = res.detections[0]
    box = d.location_data.relative_bounding_box

    x = int(box.xmin * w)
    y = int(box.ymin * h)
    w2 = int(box.width * w)
    h2 = int(box.height * h)

    x = max(0, x); y = max(0, y)
    return gray[y:y+h2, x:x+w2]

def load_data(ds: Path):
    faces, labels, paths = [], [], []
    for imgp in iter_images(ds):
        lbl = label_from_name(imgp)
        if not lbl:
            continue

        try:
            g = np.array(Image.open(imgp).convert("L"), dtype="uint8")
            g = cv2.equalizeHist(g)

            roi = detect_with_mediapipe(g)
            if roi is None:
                roi = g

            roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)

            faces.append(roi)
            labels.append(lbl)
            paths.append(str(imgp))

        except Exception as e:
            print(f"[Warn] skip {imgp}: {e}")

    return faces, labels, paths

def encode_labels(names):
    uniq = sorted(set(names))
    name_to_id = {n: i for i, n in enumerate(uniq)}
    ids = np.array([name_to_id[n] for n in names], np.int32)
    return ids, name_to_id

def stratified_split(ids, frac):
    by = defaultdict(list)
    for i, l in enumerate(ids):
        by[int(l)].append(i)

    tr, va = [], []
    for label, idxs in by.items():
        idxs = idxs[:]
        random.shuffle(idxs)
        nv = int(len(idxs) * frac)
        va += idxs[:nv]
        tr += idxs[nv:]

    return tr, va

def train_lbph(faces, ids, idxs):
    recog = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    X = [faces[i] for i in idxs]
    y = np.array([ids[i] for i in idxs], np.int32)
    recog.train(X, y)
    return recog

def simple_eval(recog, faces, ids, paths, threshold, num_labels):
    unknown = num_labels
    cm = np.zeros((num_labels, num_labels + 1), dtype=int)
    correct = unk = 0
    unk_rows = []

    for img, true_id, pth in zip(faces, ids, paths):
        pred_id, dist = recog.predict(img)
        if dist > threshold:
            cm[true_id, unknown] += 1
            unk += 1
            unk_rows.append((pth, dist))
        else:
            cm[true_id, pred_id] += 1
            if pred_id == true_id:
                correct += 1

    acc = correct / len(faces) if len(faces) else 0.0
    return acc, unk, cm, unk_rows

def print_confusion(cm, id_to_name):
    names = [id_to_name[i] for i in range(len(id_to_name))]
    print("\nConfusion matrix (rows=true, cols=pred, last=Unknown):")
    header = ["true\\pred"] + names + ["Unknown"]
    colw = max(8, max(len(h) for h in header))
    print(" ".join(h.ljust(colw) for h in header))
    for i, row in enumerate(cm):
        cells = [names[i].ljust(colw)] + [str(v).ljust(colw) for v in row]
        print(" ".join(cells))

# -------- Main --------
def main():
    random.seed(42)

    if not DATASET.exists():
        raise SystemExit(f"[Error] Dataset not found: {DATASET}")

    faces, names, paths = load_data(DATASET)
    print(f"[info] loaded {len(faces)} samples")
    if not faces:
        raise SystemExit("[Error] No usable images.")

    print("[info] per-label counts:", dict(Counter(names)))

    ids, name_to_id = encode_labels(names)
    id_to_name = {v: k for k, v in name_to_id.items()}

    if args.val_split > 0:
        tr_idx, va_idx = stratified_split(ids, args.val_split)
        recog = train_lbph(faces, ids, tr_idx)

        Xv = [faces[i] for i in va_idx]
        yv = [int(ids[i]) for i in va_idx]
        pv = [paths[i] for i in va_idx]

        acc, unk_count, cm, unk_rows = simple_eval(
            recog, Xv, yv, pv, args.threshold, len(name_to_id)
        )

        print(f"\n[VAL] acc={acc:.3f}, unknown={unk_count}")
        print_confusion(cm, id_to_name)

        if args.unknown_csv and unk_rows:
            out = Path(args.unknown_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "dist"])
                for p, d in unk_rows:
                    w.writerow([p, f"{d:.3f}"])
            print(f"[OK] Unknown list → {out}")

        MODELS.mkdir(parents=True, exist_ok=True)
        recog.save(str(MODEL_YML))

    else:
        recog = train_lbph(faces, ids, list(range(len(faces))))
        MODELS.mkdir(parents=True, exist_ok=True)
        recog.save(str(MODEL_YML))

    with open(LABELMAP, "w") as f:
        json.dump(name_to_id, f, indent=2)

    print(f"\n[OK] Model saved → {MODEL_YML}")
    print(f"[OK] Labels → {LABELMAP}")
    print("Labels:", name_to_id)


if __name__ == "__main__":
    main()