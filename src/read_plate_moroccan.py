import cv2
import easyocr
import re

# Arabic + English reader
reader_ma = easyocr.Reader(['ar', 'en'], gpu=False)

def format_moroccan_plate(text):
    text = text.replace(" ", "")
    text = re.sub(r"[^0-9A-Za-z\u0600-\u06FF]", "", text)

    letter = ""
    idx_letter = -1

    for i, c in enumerate(text):
        if '\u0600' <= c <= '\u06FF':
            letter = c
            idx_letter = i
            break

    if letter:
        left_part = text[:idx_letter]
        right_part = text[idx_letter + 1:]

        left_digits = "".join([c for c in left_part if c.isdigit()])
        right_digits = "".join([c for c in right_part if c.isdigit()])
    else:
        digits = "".join([c for c in text if c.isdigit()])
        if len(digits) >= 3:
            left_digits = digits[:-2]
            right_digits = digits[-2:]
        else:
            left_digits = digits
            right_digits = ""

    final = f"{left_digits} {letter} {right_digits}".strip()
    return final

def read_moroccan_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) < 200:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    result = reader_ma.readtext(gray)

    if not result:
        return None

    raw = " ".join([r[1] for r in result])
    formatted = format_moroccan_plate(raw)

    if formatted.strip() == "":
        return None

    return formatted
