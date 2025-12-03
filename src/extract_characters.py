import cv2
import os
from pathlib import Path

# PATHS
PLATE_DIR = "outputs/plates"              # input: cropped plates
OUT_CHAR_DIR = "data/ocr_chars/unknown"   # output: each character

os.makedirs(OUT_CHAR_DIR, exist_ok=True)

print("Extracting characters from:", PLATE_DIR)


# Helper to sort contours (left → right)
def sort_contours(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))
    return contours, bounding_boxes


# PROCESS EACH PLATE
for plate_file in os.listdir(PLATE_DIR):
    if not plate_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    full_path = os.path.join(PLATE_DIR, plate_file)
    print("\nProcessing plate:", full_path)

    plate_img = cv2.imread(full_path)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # threshold to isolate characters
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find contours (potential characters)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours, boxes = sort_contours(contours)   # sort left to right

    char_index = 0

    for c, b in zip(contours, boxes):
        x, y, w, h = b

        # filters to remove noise
        if h < 15 or w < 10:  # too small → noise
            continue

        if h / w < 1.0:  # characters are taller than wide
            continue

        # crop character
        char_img = plate_img[y:y+h, x:x+w]

        out_path = os.path.join(OUT_CHAR_DIR, f"{Path(plate_file).stem}_c{char_index}.jpg")
        cv2.imwrite(out_path, char_img)

        print("Saved character:", out_path)
        char_index += 1

print("\nDONE: Characters saved in data/ocr_chars/unknown/")
