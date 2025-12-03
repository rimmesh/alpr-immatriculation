import os
import cv2
import easyocr
import re
from pathlib import Path

# Input: cropped plates
PLATE_DIR = "outputs/plates"

# Output: labeled characters
OUT_DIR = "data/ocr_chars"
os.makedirs(OUT_DIR, exist_ok=True)

reader = easyocr.Reader(["en"])  # English letters/numbers

print("Running OCR on plates in:", PLATE_DIR)

def clean_text(t):
    # Keep only alphanumeric characters A-Z + 0-9
    return re.sub(r"[^A-Za-z0-9]", "", t).upper()


# Iterate through plate images
for plate_file in os.listdir(PLATE_DIR):
    if not plate_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    plate_path = os.path.join(PLATE_DIR, plate_file)
    print(f"\nProcessing plate: {plate_path}")

    img = cv2.imread(plate_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OCR reading
    result = reader.readtext(gray)

    if len(result) == 0:
        print("OCR failed, skipping.")
        continue

    text = clean_text(result[0][1])
    print("OCR detected text:", text)

    # Segment characters horizontally
    h, w = gray.shape
    char_width = w // max(1, len(text))

    for i, ch in enumerate(text):
        # Create folder for character
        class_dir = os.path.join(OUT_DIR, ch)
        os.makedirs(class_dir, exist_ok=True)

        # Crop character region
        x1 = i * char_width
        x2 = (i + 1) * char_width
        char_img = img[0:h, x1:x2]

        out_path = os.path.join(class_dir, f"{Path(plate_file).stem}_{i}.jpg")
        cv2.imwrite(out_path, char_img)
        print("Saved char:", ch, "→", out_path)

print("\nDONE: Characters labeled under data/ocr_chars/")
