import os
import cv2
from ocr_engine import read_plate_easyocr

PLATES_DIR = "outputs/plates"

print("\n=== Testing OCR on all detected plates ===\n")

plate_images = sorted([
    f for f in os.listdir(PLATES_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

print(f"Found {len(plate_images)} plates\n")

for name in plate_images:
    path = os.path.join(PLATES_DIR, name)
    img = cv2.imread(path)

    if img is None:
        print(f"{name} → ❌ Image read failed")
        continue

    text, conf = read_plate_easyocr(img)

    print(f"{name} → {text} (conf={conf:.2f})")
