import cv2
import easyocr
import re
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

def read_english_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 1️⃣ Strong grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Upscale aggressively
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # 3️⃣ Contrast boost
    gray = cv2.equalizeHist(gray)

    # 4️⃣ Final binarization
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # 5️⃣ OCR
    results = reader.readtext(thresh, detail=0, paragraph=False)

    if not results:
        return None

    raw = "".join(results).upper()

    # ✅ Hard cleaning — only A-Z + 0-9
    raw = raw.replace("O", "0")
    raw = raw.replace("I", "1")
    raw = raw.replace("S", "5")
    raw = re.sub(r"[^A-Z0-9]", "", raw)

    if len(raw) < 4:
        return None

    return raw
