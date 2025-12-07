import cv2
import easyocr
import re

# Load OCR once
reader_en = easyocr.Reader(['en'], gpu=False)

def clean_english_plate(text):
    text = text.upper()

    text = text.replace("O", "0")
    text = text.replace("I", "1")
    text = text.replace("Z", "2")
    text = text.replace("S", "5")
    text = text.replace("B", "8")

    text = re.sub(r"[^A-Z0-9]", "", text)

    if len(text) < 4:
        return None

    return text

def read_english_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale small plates
    h, w = gray.shape
    if max(h, w) < 200:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    result = reader_en.readtext(gray)

    if not result:
        return None

    raw = " ".join([r[1] for r in result])
    return clean_english_plate(raw)
