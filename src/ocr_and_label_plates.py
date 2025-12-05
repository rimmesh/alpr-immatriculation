import cv2
import os
from pathlib import Path

PLATE_DIR = "outputs/plates"         # cropped plates
OUT_DIR = "data/ocr_chars"           # output for characters
os.makedirs(OUT_DIR, exist_ok=True)

def sort_contours(cnts):
    bboxes = [cv2.boundingRect(c) for c in cnts]
    cnts, bboxes = zip(*sorted(zip(cnts, bboxes), key=lambda x: x[1][0]))
    return cnts, bboxes

print("\nExtracting characters using contour detection...\n")

for file in os.listdir(PLATE_DIR):
    if not file.endswith((".jpg",".png",".jpeg")):
        continue

    path = os.path.join(PLATE_DIR, file)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, boxes = sort_contours(contours)

    index = 0
    for c, b in zip(contours, boxes):
        x, y, w, h = b

        # Remove small blobs
        if w < 15 or h < 30:
            continue

        char_crop = gray[y:y+h, x:x+w]

        # save to UNLABELED folder (you label manually later)
        save_dir = os.path.join(OUT_DIR, "unlabeled")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{Path(file).stem}_{index}.jpg")
        cv2.imwrite(save_path, char_crop)
        index += 1

print("\nDONE — Characters saved inside data/ocr_chars/unlabeled\n")
