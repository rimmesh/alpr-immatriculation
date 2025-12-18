import cv2
import numpy as np

def classify_arabic_center(img):
    """
    FAST Moroccan Arabic letter classifier
    Input: binary or grayscale image (center zone)
    Output: Arabic character or ""
    """

    if img is None:
        return ""

    # Ensure binary
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(img, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = th.shape
    aspect_ratio = w / h

    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return ""

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    x, y, cw, ch = cv2.boundingRect(cnt)
    fill_ratio = area / (cw * ch + 1e-5)

    # Count dots (small contours)
    dots = 0
    for c in contours:
        if cv2.contourArea(c) < 0.08 * area:
            dots += 1

    # ---- RULES ----

    # أ : tall vertical line
    if aspect_ratio < 0.5 and fill_ratio > 0.35:
        return "أ"

    # ب : dot below
    if dots >= 1 and fill_ratio > 0.25:
        return "ب"

    # و : small loop
    if fill_ratio > 0.45 and aspect_ratio > 0.8:
        return "و"

    # د : curved, no dots
    if fill_ratio < 0.3 and dots == 0:
        return "د"

    # هـ : loop + inner hole
    if fill_ratio > 0.4 and dots == 0:
        return "هـ"

    # ج : tail-like shape
    if aspect_ratio > 0.9 and fill_ratio < 0.3:
        return "ج"

    # fallback
    return ""
