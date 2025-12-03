import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import os
import numpy as np
from pathlib import Path
import re

# MODELS
YOLO_MODEL = "models/yolo/lp_detector_v11_best.pt"
OCR_MODEL = "models/ocr/cnn_ocr.pth"

# Load YOLO detector
detector = YOLO(YOLO_MODEL)

# Load OCR CNN
ocr_model = models.resnet18(weights=None)
ocr_model.fc = nn.Linear(512, len(os.listdir("data/ocr_chars")))
ocr_model.load_state_dict(torch.load(OCR_MODEL, map_location="cpu"))
ocr_model.eval()

# OCR transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

# Get character classes
CLASSES = sorted(os.listdir("data/ocr_chars"))


# Helper — clean OCR output
def clean_text(t):
    return re.sub(r"[^A-Za-z0-9]", "", t).upper()


# 1️⃣ Segment plate into character crops
def segment_plate(plate_img, expected_chars=6):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    char_width = w // expected_chars

    crops = []
    for i in range(expected_chars):
        x1 = i * char_width
        x2 = (i + 1) * char_width
        char_crop = gray[:, x1:x2]
        crops.append(char_crop)

    return crops


# 2️⃣ Predict a single character
def predict_char(char_img):

    img = transform(char_img).unsqueeze(0).repeat(1, 3, 1, 1)

    with torch.no_grad():
        output = ocr_model(img)
        _, pred = torch.max(output, 1)

    return CLASSES[pred.item()]


# 3️⃣ Full ALPR pipeline
def predict_plate(image_path):

    results = detector(image_path)

    full_plate_number = ""

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        im0 = r.orig_img

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = im0[y1:y2, x1:x2]

            # STEP 1: segment characters
            char_crops = segment_plate(crop)

            # STEP 2: OCR each character
            predicted_chars = []
            for ch_img in char_crops:
                pred = predict_char(ch_img)
                predicted_chars.append(pred)

            full_plate_number = "".join(predicted_chars)

            print("Predicted Plate =", full_plate_number)

    return full_plate_number


if __name__ == "__main__":
    test_img = "data/yolo/test/images/your_test.jpg"
    predict_plate(test_img)
