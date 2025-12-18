import cv2
import easyocr
import re

# =========================
# OCR INIT (Arabic + English)
# =========================
reader_ma = easyocr.Reader(['ar', 'en'], gpu=False)

# =========================
# BASIC HELPERS
# =========================
def _only_digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def _first_arabic_letter(s: str) -> str:
    for c in s:
        if '\u0600' <= c <= '\u06FF':
            return c
    return ""


# =========================
# FORMAT FALLBACK
# =========================
def format_moroccan_plate(text):
    text = text.replace(" ", "")
    text = re.sub(r"[^0-9\u0600-\u06FF]", "", text)

    letter = ""
    idx_letter = -1

    for i, c in enumerate(text):
        if '\u0600' <= c <= '\u06FF':
            letter = c
            idx_letter = i
            break

    if letter:
        left_digits = _only_digits(text[:idx_letter])
        right_digits = _only_digits(text[idx_letter + 1:])
    else:
        digits = _only_digits(text)
        left_digits = digits[:-2]
        right_digits = digits[-2:]

    return f"{left_digits} {letter} {right_digits}".strip()


# =========================
# PREPROCESS
# =========================
def _preprocess_plate(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) < 200:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    th = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31, 13
    )

    return gray, th


# =========================
# ZONE SPLIT (KEEPED)
# =========================
def _split_moroccan_zones(img):
    h, w = img.shape[:2]

    x2 = int(0.42 * w)
    x3 = int(0.60 * w)

    left = img[:, :x2]
    center = img[:, x2:x3]
    right = img[:, x3:]

    return left, center, right


# =========================
# OCR WRAPPER
# =========================
def _ocr_text(img):
    result = reader_ma.readtext(img)
    if not result:
        return ""
    return " ".join(r[1] for r in result)


# =========================
# 🔥 ARABIC LETTER SHAPE FALLBACK (NEW)
# =========================
def detect_arabic_letter_shape(center_bin):
    """
    Fixes the '1 / | / I' problem.
    Works for Moroccan plates.
    """

    inv = 255 - center_bin
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return ""

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    aspect = w / float(h)
    area_ratio = cv2.contourArea(c) / (center_bin.shape[0] * center_bin.shape[1])

    # Tall thin stroke → ا
    if aspect < 0.35 and h > 0.6 * center_bin.shape[0]:
        return "ا"

    # Wider blob → ب
    if aspect >= 0.35 and area_ratio > 0.12:
        return "ب"

    return ""


# =========================
# MAIN FUNCTION
# =========================
def read_moroccan_plate(image_path, return_debug=False):

    img = cv2.imread(image_path)
    if img is None:
        return (None, {"error": "Image not found"}) if return_debug else None

    gray, th = _preprocess_plate(img)

    left_z, center_z, right_z = _split_moroccan_zones(th)

    # OCR per zone
    left_raw = _ocr_text(left_z)
    center_raw = _ocr_text(center_z)
    right_raw = _ocr_text(right_z)

    left_digits = _only_digits(left_raw)
    right_digits = _only_digits(right_raw)

    arabic_letter = _first_arabic_letter(center_raw)

    # 🔥 FIX: fallback if OCR fails or returns "1"
    if arabic_letter == "" or arabic_letter in ["1", "|", "I"]:
        arabic_letter = detect_arabic_letter_shape(center_z)

    zone_based = f"{left_digits} {arabic_letter} {right_digits}".strip()

    # Fallback full OCR
    fallback_raw = _ocr_text(gray)
    fallback_fmt = format_moroccan_plate(fallback_raw)

    final = zone_based if zone_based.replace(" ", "") else fallback_fmt

    if not return_debug:
        return final if final else None

    debug = {
        "gray": gray,
        "enhanced": th,
        "zones": {
            "left": left_z,
            "center": center_z,
            "right": right_z
        },
        "zone_raw": {
            "left": left_raw,
            "center": center_raw,
            "right": right_raw
        },
        "fallback_raw": fallback_raw,
        "final": final
    }

    return final, debug


# =========================
# CLI TEST
# =========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python read_plate_moroccan.py image.jpg")
        exit()

    plate = read_moroccan_plate(sys.argv[1])
    print("Detected plate:", plate)
