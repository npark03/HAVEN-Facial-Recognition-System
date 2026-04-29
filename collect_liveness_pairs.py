#!/usr/bin/env python3
"""
collect_liveness_pairs.py

Purpose:
- Capture 50 pairs of images for later testing
- Show a random liveness challenge between captures
- Save all images and metadata
- DOES NOT verify motion / liveness success

Useful for:
- LBPH testing
- FaceNet testing in Colab
"""

import json
import random
import subprocess
import time
from pathlib import Path
from datetime import datetime

import cv2


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "captured_liveness_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_ROUNDS = 50
WAIT_TIME = 3.0

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

CHALLENGES = [
    "LOOK_LEFT",
    "LOOK_RIGHT",
    "LOOK_UP",
]

CHALLENGE_TEXT = {
    "LOOK_LEFT": "Look LEFT",
    "LOOK_RIGHT": "Look RIGHT",
    "LOOK_UP": "Look UP",
}


# -----------------------------
# Helpers
# -----------------------------
def capture_image(filename: str):
    """
    Captures one image using rpicam-still and saves it to OUTPUT_DIR.
    Returns the loaded OpenCV image and file path.
    """
    path = OUTPUT_DIR / filename

    cmd = [
        "rpicam-still",
        "-o", str(path),
        "--width", str(IMAGE_WIDTH),
        "--height", str(IMAGE_HEIGHT),
        "--nopreview",
        "-t", "1000",
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

    # Optional flip if your camera is mirrored
    image = cv2.flip(image, 1)

    # Re-save the flipped version so the saved file matches what you want
    cv2.imwrite(str(path), image)

    return image, path


def choose_challenge():
    return random.choice(CHALLENGES)


def run_collection():
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / f"session_{session_timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "session": session_timestamp,
        "num_rounds_requested": NUM_ROUNDS,
        "wait_time_seconds": WAIT_TIME,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "rounds": []
    }

    print("\nStarting liveness pair collection...")
    print(f"Saving to: {session_dir}")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"Wait time between images: {WAIT_TIME} seconds\n")

    for i in range(1, NUM_ROUNDS + 1):
        print("=" * 50)
        print(f"ROUND {i}/{NUM_ROUNDS}")
        print("=" * 50)

        challenge = choose_challenge()

        img1_name = f"round_{i:03d}_img1.jpg"
        img2_name = f"round_{i:03d}_img2.jpg"

        img1_path = session_dir / img1_name
        img2_path = session_dir / img2_name

        # Capture first image
        print("Capturing first image...")
        img1, temp_path1 = capture_image(img1_name)
        temp_path1.rename(img1_path)

        # Print challenge
        print("\nLIVENESS CHALLENGE")
        print("------------------")
        print(CHALLENGE_TEXT[challenge])
        print(f"\nCapturing second image in {WAIT_TIME:.1f} seconds...\n")

        time.sleep(WAIT_TIME)

        # Capture second image
        print("Capturing second image...")
        img2, temp_path2 = capture_image(img2_name)
        temp_path2.rename(img2_path)

        round_record = {
            "round": i,
            "challenge": challenge,
            "challenge_text": CHALLENGE_TEXT[challenge],
            "image_1": str(img1_path),
            "image_2": str(img2_path),
            "timestamp": datetime.now().isoformat(),
        }

        metadata["rounds"].append(round_record)

        print(f"Saved:")
        print(f"  {img1_path.name}")
        print(f"  {img2_path.name}")
        print()

    metadata_path = session_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Images saved in: {session_dir}")
    print(f"Metadata saved in: {metadata_path}")


if __name__ == "__main__":
    try:
        run_collection()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")