import os
import cv2
import glob
from ultralytics import YOLO
from pathlib import Path

# PATHS
MODEL_PATH = "models/yolo/lp_detector_v11_best.pt"     # YOLO trained model
INPUT_DIR = "data/yolo/test/images"                     # full car images
OUT_DET_DIR = "outputs/detections"                      # YOLO detections
OUT_PLATE_DIR = "outputs/plates"                        # cropped plates


# Create output folders
os.makedirs(OUT_DET_DIR, exist_ok=True)
os.makedirs(OUT_PLATE_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load all test images
image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + \
              glob.glob(os.path.join(INPUT_DIR, "*.png")) + \
              glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))

print(f"Found {len(image_paths)} images")


# PROCESS EACH IMAGE
for img_path in image_paths:
    print("\nProcessing:", img_path)
    img_name = Path(img_path).stem

    results = model(img_path, conf=0.4)

    for r in results:
        # save detection image
        det_img = r.plot()
        det_path = os.path.join(OUT_DET_DIR, f"{img_name}_det.jpg")
        cv2.imwrite(det_path, det_img)
        print("Saved detection:", det_path)

        im0 = r.orig_img
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            conf = confs[i]

            # Filter: remove low confidence
            if conf < 0.45:
                continue

            # Filter: remove small boxes
            w, h = x2 - x1, y2 - y1
            if w < 80 or h < 30:
                continue

            # Filter: aspect ratio (plates are wide)
            if w / h < 1.5:
                continue

            # Crop plate
            crop = im0[y1:y2, x1:x2]
            crop_path = os.path.join(OUT_PLATE_DIR, f"{img_name}_plate_{i}.jpg")
            cv2.imwrite(crop_path, crop)
            print("Saved plate crop:", crop_path)

print("\nDONE: All plate crops saved to outputs/plates/")
