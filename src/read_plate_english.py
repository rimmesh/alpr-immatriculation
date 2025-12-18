import cv2
import easyocr
import re
from typing import Tuple, Optional, Dict, Any

# English OCR only
reader_en = easyocr.Reader(['en'], gpu=False)

# -----------------------------------
# ENHANCEMENT (KEEP THIS)
# -----------------------------------
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
    return gray, th


# -----------------------------------
# CHARACTER SEGMENTATION (VISUAL ONLY)
# -----------------------------------
def segment_characters(binary_img):
    H, W = binary_img.shape[:2]
    inv = 255 - binary_img

    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / float(h)

        if h < 0.4 * H:
            continue
        if h > 0.95 * H:
            continue
        if aspect < 0.15 or aspect > 1.2:
            continue
        if area < 0.002 * (W * H):
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    debug = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return boxes, debug


# -----------------------------------
# MAIN PIPELINE WITH DEBUG
# -----------------------------------
def read_english_plate_debug(
    image_path: str
) -> Tuple[Optional[str], Dict[str, Any]]:

    img = cv2.imread(image_path)
    if img is None:
        return None, {"error": "Image not found"}

    # 1️⃣ Enhance
    gray, th = enhance_plate(img)

    # 2️⃣ Character segmentation (VISUAL ONLY)
    boxes, char_debug = segment_characters(th)

    # 3️⃣ OCR ON ENHANCED IMAGE (IMPORTANT)
    ocr = reader_en.readtext(th, detail=1, paragraph=False)

    # sort left → right
    def left_x(item):
        return min(p[0] for p in item[0])

    ocr = sorted(ocr, key=left_x)

    raw = "".join([r[1] for r in ocr]).upper()
    clean = "".join(re.findall(r"[A-Z0-9]", raw))
    final = clean if clean else None

    debug = {
        "gray": gray,
        "threshold": th,
        "char_boxes": char_debug,
        "characters_count": len(boxes),
        "raw_ocr": raw,
    }

    return final, debug


# -----------------------------------
# PUBLIC FUNCTION (USED BY INTERFACE)
# -----------------------------------
def read_english_plate(
    image_path: str,
    return_debug: bool = False
):
    result, debug = read_english_plate_debug(image_path)

    if return_debug:
        return result, debug

    return result
