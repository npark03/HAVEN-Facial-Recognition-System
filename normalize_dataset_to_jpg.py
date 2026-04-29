#!/home/haven/haven-security/venv/bin/python

import os
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

# Enable HEIC/HEIF support
register_heif_opener()

DATASET_DIR = Path("/home/haven/haven-security/dataset")

# Extensions that we will try to convert
CONVERTIBLE_EXTENSIONS = {
    ".png",
    ".bmp",
    ".webp",
    ".heic",
    ".heif",
    ".tiff",
    ".tif"
}

# Extensions already acceptable
JPG_EXTENSIONS = {".jpg", ".jpeg"}

def convert_to_jpg(image_path: Path) -> bool:
    """
    Convert one image to JPG.
    Returns True if conversion succeeded, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            # Convert modes like RGBA, P, etc. to RGB for JPEG
            img = img.convert("RGB")

            output_path = image_path.with_suffix(".jpg")

            # Avoid overwriting if a JPG already exists
            if output_path.exists():
                stem = image_path.stem
                parent = image_path.parent
                counter = 1
                while True:
                    candidate = parent / f"{stem}_converted_{counter}.jpg"
                    if not candidate.exists():
                        output_path = candidate
                        break
                    counter += 1

            img.save(output_path, "JPEG", quality=95)
            print(f"[CONVERTED] {image_path} -> {output_path}")
            return True

    except Exception as e:
        print(f"[FAILED] {image_path} -> {e}")
        return False

def scan_dataset(dataset_dir: Path):
    """
    Walk through dataset folder and convert non-JPG images where possible.
    """
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset folder not found: {dataset_dir}")
        return

    total_files = 0
    already_jpg = 0
    converted = 0
    skipped = 0
    failed = 0

    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            total_files += 1
            file_path = Path(root) / filename
            ext = file_path.suffix.lower()

            if ext in JPG_EXTENSIONS:
                already_jpg += 1
                print(f"[OK] {file_path}")
                continue

            if ext in CONVERTIBLE_EXTENSIONS:
                success = convert_to_jpg(file_path)
                if success:
                    converted += 1
                else:
                    failed += 1
                continue

            skipped += 1
            print(f"[SKIPPED] Unsupported file type: {file_path}")

    print("\n===== SUMMARY =====")
    print(f"Total files checked: {total_files}")
    print(f"Already JPG/JPEG:    {already_jpg}")
    print(f"Converted to JPG:    {converted}")
    print(f"Skipped:             {skipped}")
    print(f"Failed:              {failed}")

if __name__ == "__main__":
    scan_dataset(DATASET_DIR)