#!/usr/bin/env python3
"""
run_liveness_check_lbph.py

Flow:
1. Capture first image
2. Detect face in first image
3. Print random liveness challenge
4. Wait
5. Capture second image
6. Detect face in second image
7. Compare face movement for liveness
8. If liveness passes, use FIRST image for LBPH authorization

Uses:
- rpicam-still
- OpenCV Haar cascade
- LBPH model from model/lbph/trainer.yml
"""

import json
import random
import subprocess
import time
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
CAPTURED_DIR = BASE_DIR / "captured"
LBPH_DIR = BASE_DIR / "model" / "lbph"

CAPTURED_DIR.mkdir(parents=True, exist_ok=True)

TRAINER_FILE = LBPH_DIR / "trainer.yml"
LABELS_FILE = LBPH_DIR / "labels.json"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

WAIT_TIME = 3.0

# Tune these based on your real testing
HORIZONTAL_THRESHOLD = 45
VERTICAL_THRESHOLD = 25

# Lower confidence is better in LBPH.
# You will likely need to tune this.
LBPH_CONFIDENCE_THRESHOLD = 70.0


def load_face_detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {CASCADE_PATH}")
    return detector


def load_lbph():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError as e:
        raise RuntimeError(
            "cv2.face is not available. Install opencv-contrib-python."
        ) from e

    if not TRAINER_FILE.exists():
        raise FileNotFoundError(f"Missing LBPH model: {TRAINER_FILE}")

    recognizer.read(str(TRAINER_FILE))
    return recognizer


def load_labels():
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_FILE}")

    with open(LABELS_FILE, "r") as f:
        return json.load(f)


def capture_image(filename: str):
    path = CAPTURED_DIR / filename

    cmd = [
        "rpicam-still",
        "-o", str(path),
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

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read captured image: {path}")

    # Fix mirrored image
    image = cv2.flip(image, 1)

    return image, path


def choose_challenge():
    return random.choice(["LOOK_LEFT", "LOOK_RIGHT", "LOOK_UP"])


def challenge_text(challenge: str) -> str:
    mapping = {
        "LOOK_LEFT": "Look LEFT",
        "LOOK_RIGHT": "Look RIGHT",
        "LOOK_UP": "Look UP",
    }
    return mapping[challenge]


def detect_largest_face_box(image_bgr, detector):
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


def face_center(face_box):
    x, y, w, h = face_box
    return (x + w / 2.0, y + h / 2.0)


def check_liveness(challenge, box1, box2):
    c1 = face_center(box1)
    c2 = face_center(box2)

    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]

    if challenge == "LOOK_LEFT":
        passed = dx <= -HORIZONTAL_THRESHOLD
    elif challenge == "LOOK_RIGHT":
        passed = dx >= HORIZONTAL_THRESHOLD
    elif challenge == "LOOK_UP":
        passed = dy <= -VERTICAL_THRESHOLD
    else:
        passed = False

    debug = {
        "movement_x": round(dx, 2),
        "movement_y": round(dy, 2),
        "face1_center": [round(c1[0], 2), round(c1[1], 2)],
        "face2_center": [round(c2[0], 2), round(c2[1], 2)],
    }

    return passed, debug


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

    face = cv2.resize(face, (200, 200))
    return face


def authorize_first_image(image_bgr, detector, recognizer, labels):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    face_box = detect_largest_face_box(image_bgr, detector)
    if face_box is None:
        return {
            "authorized": False,
            "reason": "no_face_for_authorization",
            "identity": None,
            "confidence": None,
            "threshold": LBPH_CONFIDENCE_THRESHOLD,
        }

    face = crop_face(gray, face_box)

    label_id, confidence = recognizer.predict(face)

    identity = labels.get(str(label_id), "unknown")
    authorized = confidence <= LBPH_CONFIDENCE_THRESHOLD

    return {
        "authorized": bool(authorized),
        "reason": "ok" if authorized else "face_not_recognized",
        "identity": identity if authorized else "unknown",
        "predicted_label_id": int(label_id),
        "confidence": round(float(confidence), 4),
        "threshold": LBPH_CONFIDENCE_THRESHOLD,
    }


def run_liveness_check_lbph():
    detector = load_face_detector()
    recognizer = load_lbph()
    labels = load_labels()

    print("\nCapturing first image...")
    img1, path1 = capture_image("liveness_1.jpg")

    box1 = detect_largest_face_box(img1, detector)
    if box1 is None:
        return {
            "liveness_passed": False,
            "reason": "no_face_first_image",
            "authorized": False,
            "identity": None,
            "capture_1": str(path1),
        }

    challenge = choose_challenge()

    print("\nLIVENESS CHALLENGE")
    print("==================")
    print(challenge_text(challenge))
    print(f"\nCapturing second image in {WAIT_TIME:.1f} seconds...\n")

    time.sleep(WAIT_TIME)

    img2, path2 = capture_image("liveness_2.jpg")

    box2 = detect_largest_face_box(img2, detector)
    if box2 is None:
        return {
            "liveness_passed": False,
            "reason": "no_face_second_image",
            "challenge": challenge,
            "authorized": False,
            "identity": None,
            "capture_1": str(path1),
            "capture_2": str(path2),
        }

    liveness_passed, debug = check_liveness(challenge, box1, box2)

    if not liveness_passed:
        return {
            "liveness_passed": False,
            "reason": "challenge_failed",
            "challenge": challenge,
            "authorized": False,
            "identity": None,
            "capture_1": str(path1),
            "capture_2": str(path2),
            "debug": debug,
        }

    auth_result = authorize_first_image(img1, detector, recognizer, labels)

    return {
        "liveness_passed": True,
        "reason": "ok",
        "challenge": challenge,
        "capture_1": str(path1),
        "capture_2": str(path2),
        "debug": debug,
        "authorized": auth_result["authorized"],
        "identity": auth_result["identity"],
        "confidence": auth_result["confidence"],
        "predicted_label_id": auth_result["predicted_label_id"],
        "auth_reason": auth_result["reason"],
        "auth_threshold": auth_result["threshold"],
    }


if __name__ == "__main__":
    try:
        result = run_liveness_check_lbph()
    except Exception as e:
        result = {
            "liveness_passed": False,
            "authorized": False,
            "reason": f"error: {e}",
        }

    print("\nRESULT")
    print("======")
    print(json.dumps(result, indent=2))