import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/yolo/lp_detector_v11_best.pt"
OUT_PLATE_DIR = "outputs/plates"
os.makedirs(OUT_PLATE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

def detect_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not readable")
        return None

    results = model(img, conf=0.40)

    if len(results) == 0 or results[0].boxes is None:
        print(" No plate detected")
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()

    # ✅ Pick the LARGEST box (most reliable)
    best_box = None
    best_area = 0

    for b in boxes:
        x1, y1, x2, y2 = map(int, b[:4])
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box

    # ✅ EXPAND box to fix half-crop Moroccan issue
    h, w, _ = img.shape
    expand_x = int((x2 - x1) * 0.45)
    expand_y = int((y2 - y1) * 0.35)

    x1 = max(0, x1 - expand_x)
    y1 = max(0, y1 - expand_y)
    x2 = min(w, x2 + expand_x)
    y2 = min(h, y2 + expand_y)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        print("Empty crop")
        return None

    crop_path = os.path.join(OUT_PLATE_DIR, f"plate_{os.path.basename(image_path)}")
    cv2.imwrite(crop_path, crop)

    return crop_path,crop
