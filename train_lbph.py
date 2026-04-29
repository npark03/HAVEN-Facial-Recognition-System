#!/usr/bin/env python3
"""
train_lbph.py

Train an LBPH face recognizer from:
    dataset/<person_name>/*.jpg

Saves:
    model/lbph/trainer.yml
    model/lbph/labels.json
    model/lbph/metadata.json
"""

import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model" / "lbph"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAINER_FILE = MODEL_DIR / "trainer.yml"
LABELS_FILE = MODEL_DIR / "labels.json"
METADATA_FILE = MODEL_DIR / "metadata.json"

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_face_detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {CASCADE_PATH}")
    return detector


def detect_largest_face(gray_image, detector):
    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]


def crop_face(gray_image, face_box, padding=0.15):
    x, y, w, h = face_box
    img_h, img_w = gray_image.shape[:2]

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    face = gray_image[y1:y2, x1:x2]
    if face.size == 0:
        raise ValueError("Face crop was empty")

    # Normalize to a consistent size for LBPH
    face = cv2.resize(face, (200, 200))
    return face


def main():
    detector = load_face_detector()

    # LBPH is in opencv-contrib-python, not plain opencv-python
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
    except AttributeError as e:
        raise RuntimeError(
            "cv2.face is not available. Install opencv-contrib-python."
        ) from e

    faces = []
    label_ids = []
    label_map = {}
    next_label_id = 0

    print("\nScanning dataset...\n")

    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name

        if person_name not in label_map:
            label_map[person_name] = next_label_id
            next_label_id += 1

        label_id = label_map[person_name]
        used_count = 0

        print(f"Processing {person_name}")

        for image_path in sorted(person_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  [WARN] Could not read: {image_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_box = detect_largest_face(gray, detector)

            if face_box is None:
                print(f"  [WARN] No face found: {image_path}")
                continue

            try:
                face = crop_face(gray, face_box)
                faces.append(face)
                label_ids.append(label_id)
                used_count += 1
            except Exception as e:
                print(f"  [WARN] Failed processing {image_path}: {e}")

        print(f"  Used {used_count} image(s)\n")

    if not faces:
        raise RuntimeError("No usable faces found in dataset.")

    recognizer.train(faces, np.array(label_ids))
    recognizer.write(str(TRAINER_FILE))

    # Save labels as id -> name
    id_to_name = {str(v): k for k, v in label_map.items()}

    with open(LABELS_FILE, "w") as f:
        json.dump(id_to_name, f, indent=2)

    metadata = {
        "num_people": len(label_map),
        "num_training_images": len(faces),
        "image_size": [200, 200],
        "lbph_params": {
            "radius": 1,
            "neighbors": 8,
            "grid_x": 8,
            "grid_y": 8
        }
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.\n")
    print(f"Saved model:   {TRAINER_FILE}")
    print(f"Saved labels:  {LABELS_FILE}")
    print(f"Saved metadata:{METADATA_FILE}")


if __name__ == "__main__":
    main()