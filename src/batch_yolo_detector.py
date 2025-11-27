from ultralytics import YOLO   # run your YOLOv11 (or v8) model
import cv2                     # image writing & cropping
import os                      # folders & paths
import glob                    # listing files in a folder
from pathlib import Path       # easy filename handling
import csv                     # writing results.csv
import easyocr                 # OCR on cropped plates
import re                      # regex cleanup of OCR text


# PATHS
MODEL_PATH   = "models/yolo/lp_detector_v11_best.pt"
IMAGE_DIR    = "data/yolo/test/images"
OUT_DET_DIR  = "outputs/detections"
OUT_PLATE_DIR = "outputs/plates"
RESULTS_CSV  = "outputs/results.csv"

# Make sure output folders exist
os.makedirs(OUT_DET_DIR, exist_ok=True)
os.makedirs(OUT_PLATE_DIR, exist_ok=True)

# Load YOLO model + EasyOCR reader
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])   # English for alphanumeric plates

# Open CSV file
csv_file = open(RESULTS_CSV, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["image", "plate_index", "x1", "y1", "x2", "y2", "confidence", "clean_text"])

# Collect all images in IMAGE_DIR
image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) \
             + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) \
             + glob.glob(os.path.join(IMAGE_DIR, "*.png"))

print(f"Found {len(image_paths)} images")


# PROCESS EACH IMAGE (YOLO → FILTER → CROP → OCR → CLEAN)
for img_path in image_paths:
    print(f"\nProcessing: {img_path}")
    img_name = Path(img_path).stem

    # set YOLO confidence threshold 
    results = model(img_path, conf=0.40)

    for r in results:
        # Save YOLO image with bounding boxes
        im_plot = r.plot()
        det_out_path = os.path.join(OUT_DET_DIR, f"{img_name}_det.jpg")
        cv2.imwrite(det_out_path, im_plot)
        print("Saved detection:", det_out_path)

        im0 = r.orig_img
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()   # NEW: confidence values

        # LOOP through all detected objects
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            conf = confs[i]

            #  Ignore LOW-CONFIDENCE detections
            #    removes weak detections like random text on buses
            if conf < 0.45:
                continue

            # Ignore VERY SMALL boxes
            #    plates must have a minimum size or OCR will fail
            w = x2 - x1
            h = y2 - y1
            if w < 80 or h < 30:   # aggressive filtering
                continue

            #  Check aspect ratio (plates are wide)
            aspect_ratio = w / h
            if aspect_ratio < 1.5:   # too square = not a plate
                continue

            # Crop the detected plate
            plate_crop = im0[y1:y2, x1:x2]

            crop_out_path = os.path.join(OUT_PLATE_DIR, f"{img_name}_plate_{i}.jpg")
            cv2.imwrite(crop_out_path, plate_crop)
            print("Saved plate crop:", crop_out_path)

            # OCR WITH PREPROCESSING
            # Convert crop to grayscale
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            # Apply threshold to enhance text
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Perform OCR on thresholded image
            ocr_result = reader.readtext(th)

            if len(ocr_result) > 0:
                raw_text = ocr_result[0][1]
            else:
                raw_text = ""


            # CLEAN OCR text with regex
            #    keeps only A-Z and 0-9
            clean_text = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

            print("Recognized text:", clean_text)

            # Save in CSV
            writer.writerow([img_name, i, x1, y1, x2, y2, float(conf), clean_text])


csv_file.close()
print("\nAll done. Results saved in", RESULTS_CSV)