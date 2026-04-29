#!/usr/bin/env python3
"""
run_access_check.py

Flow:
1. Capture image with rpicam-still
2. Save image to captured/latest.jpg
3. Detect the largest face
4. Generate FaceNet embedding
5. Compare against premade embeddings
6. Return authorized true/false

Expected files:
- model/facenet/embeddings.pkl
- captured/
"""

import json
import pickle
import subprocess
from pathlib import Path

import cv2
import numpy as np
from keras_facenet import FaceNet


BASE_DIR = Path(__file__).resolve().parent
CAPTURED_DIR = BASE_DIR / "captured"
MODEL_DIR = BASE_DIR / "model" / "facenet"

CAPTURED_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_FILE = MODEL_DIR / "embeddings.pkl"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

THRESHOLD = 0.90


def load_embeddings():
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Missing embeddings file: {EMBEDDINGS_FILE}")

    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)

    if not data:
        raise ValueError("embeddings.pkl is empty")

    return data


def capture_image():
    output_path = CAPTURED_DIR / "latest.jpg"

    cmd = [
        "rpicam-still",
        "-o", str(output_path),
        "--width", "1280",
        "--height", "720",
        "--nopreview",
        "-t", "1000"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "Camera capture failed.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    image = cv2.imread(str(output_path))
    if image is None:
        raise RuntimeError(f"Failed to read captured image: {output_path}")

    return image, output_path


def detect_largest_face(image_bgr, detector):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]


def crop_face(image_bgr, face_box, padding=0.20):
    x, y, w, h = face_box
    img_h, img_w = image_bgr.shape[:2]

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    face = image_bgr[y1:y2, x1:x2]
    if face.size == 0:
        raise ValueError("Face crop was empty")

    return face


def get_embedding(face_bgr, embedder):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    embedding = embedder.embeddings([face_rgb])[0]
    embedding = np.asarray(embedding, dtype=np.float32)

    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Embedding norm was zero")

    return embedding / norm


def compare_embedding(query_embedding, known_embeddings):
    best_name = None
    best_distance = float("inf")

    for name, stored_embedding in known_embeddings.items():
        known_embedding = np.asarray(stored_embedding, dtype=np.float32)
        distance = float(np.linalg.norm(query_embedding - known_embedding))

        if distance < best_distance:
            best_distance = distance
            best_name = name

    return best_name, best_distance


def run_access_check():
    known_embeddings = load_embeddings()

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {CASCADE_PATH}")

    embedder = FaceNet()

    image, capture_path = capture_image()

    face_box = detect_largest_face(image, detector)
    if face_box is None:
        return {
            "authorized": False,
            "reason": "no_face_detected",
            "identity": None,
            "distance": None,
            "capture_path": str(capture_path)
        }

    face = crop_face(image, face_box)
    query_embedding = get_embedding(face, embedder)

    identity, distance = compare_embedding(query_embedding, known_embeddings)
    authorized = distance < THRESHOLD

    return {
        "authorized": bool(authorized),
        "reason": "ok" if authorized else "face_not_recognized",
        "identity": identity if authorized else "unknown",
        "distance": round(distance, 4),
        "threshold": THRESHOLD,
        "capture_path": str(capture_path)
    }


if __name__ == "__main__":
    try:
        result = run_access_check()
    except Exception as e:
        result = {
            "authorized": False,
            "reason": f"error: {e}",
            "identity": None,
            "distance": None
        }

    print(json.dumps(result, indent=2))