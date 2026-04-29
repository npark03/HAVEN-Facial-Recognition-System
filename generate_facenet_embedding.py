"""
Generate FaceNet embeddings from dataset images.

Reads images from:
    dataset/<person_name>/*.jpg

Creates:
    model/facenet/embeddings.pkl
    model/facenet/labels.pkl
    model/facenet/metadata.json

Each person gets ONE averaged embedding.
"""

import os
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from keras_facenet import FaceNet

# =====================
# paths
# =====================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model" / "facenet"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# face detector
# =====================

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# =====================
# facenet model
# =====================

embedder = FaceNet()

# =====================
# helper functions
# =====================

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # take largest face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

    x, y, w, h = faces[0]

    padding = 0.2

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)

    x2 = x + w + pad_w
    y2 = y + h + pad_h

    face = image[y1:y2, x1:x2]

    return face


def get_embedding(face_image):

    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    embeddings = embedder.embeddings([face_rgb])

    emb = embeddings[0]

    emb = emb / np.linalg.norm(emb)

    return emb


# =====================
# build embeddings
# =====================

embeddings = {}
labels = []

print("\nScanning dataset folder...\n")

for person_dir in DATASET_DIR.iterdir():

    if not person_dir.is_dir():
        continue

    person_name = person_dir.name

    person_embeddings = []

    print(f"Processing {person_name}")

    for image_path in person_dir.glob("*"):

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read {image_path}")
            continue

        face = detect_face(image)

        if face is None:
            print(f"No face found in {image_path}")
            continue

        try:
            emb = get_embedding(face)
            person_embeddings.append(emb)

        except Exception as e:
            print(f"Embedding failed for {image_path}")
            print(e)

    if len(person_embeddings) == 0:
        print(f"No usable images for {person_name}")
        continue

    avg_embedding = np.mean(person_embeddings, axis=0)

    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    embeddings[person_name] = avg_embedding.tolist()

    labels.append(person_name)

    print(f"{person_name} -> {len(person_embeddings)} images used\n")


# =====================
# save files
# =====================

with open(MODEL_DIR / "embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

with open(MODEL_DIR / "labels.pkl", "wb") as f:
    pickle.dump(labels, f)

metadata = {
    "num_people": len(labels),
    "people": labels,
    "embedding_size": len(list(embeddings.values())[0]) if embeddings else 0
}

with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nDone\n")

print("Saved to:")
print(MODEL_DIR / "embeddings.pkl")
print(MODEL_DIR / "labels.pkl")
print(MODEL_DIR / "metadata.json")