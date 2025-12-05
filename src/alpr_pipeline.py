import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import os
import numpy as np

YOLO_MODEL = "models/yolo/lp_detector_v11_best.pt"
OCR_MODEL  = "models/ocr/cnn_ocr.pth"

detector = YOLO(YOLO_MODEL)

CLASSES = sorted(os.listdir("data/ocr_chars"))

ocr_model = models.resnet18(weights=None)
ocr_model.fc = nn.Linear(512, len(CLASSES))
ocr_model.load_state_dict(torch.load(OCR_MODEL, map_location="cpu"))
ocr_model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

def predict_char(gray_char):
    img = transform(gray_char).unsqueeze(0).repeat(1,3,1,1)
    with torch.no_grad():
        out = ocr_model(img)
        _, pred = torch.max(out,1)
    return CLASSES[pred.item()]

def predict_plate(image_path):

    results = detector(image_path)
    final_plate = ""

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        im0 = r.orig_img

        for box in boxes:
            x1,y1,x2,y2 = map(int,box[:4])
            crop = im0[y1:y2,x1:x2]

            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            h,w = gray.shape

            char_w = w // 6  # assume 6-character plate

            predicted = []
            for i in range(6):
                x1 = i*char_w
                x2 = (i+1)*char_w
                char_crop = gray[:,x1:x2]
                predicted.append(predict_char(char_crop))

            final_plate = "".join(predicted)
            print("Predicted Plate:", final_plate)

    return final_plate

if __name__ == "__main__":
    predict_plate("data/yolo/test/images/your_test.jpg")
