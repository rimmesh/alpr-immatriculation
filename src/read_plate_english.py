import cv2
import easyocr
import re

# English OCR only (most stable)
reader_en = easyocr.Reader(['en'], gpu=False)

def enhance_plate(img):
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

    th = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        25, 15
    )

    return th

def read_english_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found")
        return None

    img = enhance_plate(img)

    results = reader_en.readtext(img, detail=0)

    raw = " ".join(results).upper()

    clean = "".join(re.findall(r"[A-Z0-9]", raw))

    if clean == "":
        return None

    return clean
