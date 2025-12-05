import os
import cv2
import glob
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "models/yolo/lp_detector_v11_best.pt"
INPUT_DIRS = [
    "data/yolo/test/images",
    "data/raw/moroccan/DATA"
]
OUT_DET_DIR = "outputs/detections"
OUT_PLATE_DIR = "outputs/plates"

os.makedirs(OUT_DET_DIR, exist_ok=True)
os.makedirs(OUT_PLATE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

image_paths = []

for d in INPUT_DIRS:
    image_paths += glob.glob(os.path.join(d, "*.jpg"))
    image_paths += glob.glob(os.path.join(d, "*.png"))
    image_paths += glob.glob(os.path.join(d, "*.jpeg"))


print(f"Found {len(image_paths)} images")

for img_path in image_paths:
    print("\nProcessing:", img_path)
    img_name = Path(img_path).stem

    results = model(img_path, conf=0.4)

    for r in results:
        det_img = r.plot()
        cv2.imwrite(f"{OUT_DET_DIR}/{img_name}_det.jpg", det_img)

        im0 = r.orig_img
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            conf = confs[i]

            if conf < 0.45: continue
            w, h = x2 - x1, y2 - y1
            if w < 80 or h < 30: continue
            if w/h < 1.5: continue

            crop = im0[y1:y2, x1:x2]
            cv2.imwrite(f"{OUT_PLATE_DIR}/{img_name}_plate_{i}.jpg", crop)

print("\nDONE.")
