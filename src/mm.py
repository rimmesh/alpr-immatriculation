import cv2
import numpy as np
import easyocr
import re

# =====================================
# OCR INIT (Arabic + English)
# =====================================
reader_ma = easyocr.Reader(["ar", "en"], gpu=False)

# Moroccan plate Arabic letters (common official set)
MOROCCAN_AR_LETTERS = set("ابجدهوط")

# =====================================
# HELPERS
# =====================================
def _only_digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())

def _clean_text_keep_ar_digits(s: str) -> str:
    # keep digits + arabic letters only
    return re.sub(r"[^0-9\u0600-\u06FF]", "", s)

def _extract_valid_arabic(s: str) -> str:
    s = re.sub(r"[^\u0600-\u06FF]", "", s)
    for c in s:
        if c in MOROCCAN_AR_LETTERS:
            return c
    return ""

def _ocr_text(img, allowlist=None) -> str:
    try:
        res = reader_ma.readtext(img, detail=0, allowlist=allowlist)
    except TypeError:
        # older easyocr might not support allowlist in some builds
        res = reader_ma.readtext(img, detail=0)
    if not res:
        return ""
    return " ".join(res)

# =====================================
# PREPROCESS
# =====================================
def _preprocess_plate(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) < 240:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # denoise a bit
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    th = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31, 13
    )
    return gray, th

# =====================================
# SPLIT BY '|' DETECTION
# =====================================
def _find_separator_columns(th_bin):
    """
    Detect the two vertical separators '|' by looking at column black-pixel density.
    Returns (x_left_sep, x_right_sep) or (None, None)
    """
    # In th_bin: black text = 0, white bg = 255
    black = (th_bin == 0).astype(np.uint8)
    col_sum = black.sum(axis=0).astype(np.float32)  # black pixels per column

    # Smooth the projection (helps stabilize peaks)
    k = max(7, (th_bin.shape[1] // 120) * 2 + 1)  # odd kernel size
    kernel = np.ones(k, dtype=np.float32) / k
    col_smooth = np.convolve(col_sum, kernel, mode="same")

    # Build a threshold: peaks significantly above average
    mu = float(col_smooth.mean())
    sigma = float(col_smooth.std()) + 1e-6
    thr = mu + 2.2 * sigma

    # Candidate columns
    mask = col_smooth > thr

    # Group contiguous candidate columns into segments
    segments = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            end = i - 1
            segments.append((start, end))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))

    # Score each segment by peak value and slenderness
    scored = []
    for (a, b) in segments:
        width = b - a + 1
        peak = float(col_smooth[a:b + 1].max())
        # separators should be relatively thin
        if width <= max(25, th_bin.shape[1] * 0.04):
            center = (a + b) // 2
            scored.append((peak, center, a, b))

    if len(scored) < 2:
        return None, None

    # Pick the best two separators, then order left->right
    scored.sort(key=lambda x: x[0], reverse=True)
    best_two = sorted(scored[:2], key=lambda x: x[1])
    return best_two[0][1], best_two[1][1]

def _split_moroccan_zones_by_separators(th_bin):
    h, w = th_bin.shape[:2]
    s1, s2 = _find_separator_columns(th_bin)

    if s1 is None or s2 is None or s2 <= s1 + 10:
        # Fallback to old ratio split if detection fails
        x2 = int(0.42 * w)
        x3 = int(0.60 * w)
        return th_bin[:, :x2], th_bin[:, x2:x3], th_bin[:, x3:], (None, None)

    # expand around separator a little (so the '|' doesn’t contaminate OCR)
    pad = int(0.015 * w)
    a = max(0, s1 - pad)
    b = min(w, s1 + pad)
    c = max(0, s2 - pad)
    d = min(w, s2 + pad)

    left = th_bin[:, :a]
    center = th_bin[:, b:c]
    right = th_bin[:, d:]

    return left, center, right, (s1, s2)

# =====================================
# ARABIC LETTER SHAPE FALLBACK
# =====================================
def _detect_arabic_letter_shape(center_bin):
    """
    Heuristic fallback for confusing cases where OCR returns | / 1 / I.
    Distinguish mainly between ا and ب (common in your samples).
    """
    # invert: letters become white on black for contour/cc analysis
    inv = 255 - center_bin
    # clean small noise
    inv = cv2.medianBlur(inv, 3)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)

    if num_labels <= 1:
        return ""

    # remove background label 0
    comps = []
    H, W = inv.shape[:2]
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 0.002 * H * W:
            continue
        comps.append((area, i, x, y, w, h))

    if not comps:
        return ""

    comps.sort(reverse=True, key=lambda t: t[0])
    main = comps[0]
    area, i, x, y, w, h = main

    aspect = w / float(h + 1e-6)

    # Tall thin -> ا
    if aspect < 0.38 and h > 0.55 * H:
        return "ا"

    # Look for a small dot component below/near main body -> ب
    main_cx, main_cy = centroids[i]
    for (a2, j, x2, y2, w2, h2) in comps[1:]:
        cx, cy = centroids[j]
        # dot usually smaller and below-ish
        if a2 < 0.25 * area and cy > main_cy and abs(cx - main_cx) < 0.35 * W:
            return "ب"

    # If it is wide-ish and has enough fill, likely ب
    fill_ratio = area / float(H * W)
    if aspect >= 0.38 and fill_ratio > 0.08:
        return "ب"

    return ""

# =====================================
# MAIN
# =====================================
def read_moroccan_plate(image_path, return_debug=False):
    img = cv2.imread(image_path)
    if img is None:
        return (None, {"error": "Image not found", "zone_raw": {"left": "", "center": "", "right": ""}}) if return_debug else None

    gray, th = _preprocess_plate(img)

    left_z, center_z, right_z, seps = _split_moroccan_zones_by_separators(th)

    # OCR per zone (restrict digits zones)
    left_raw = _ocr_text(left_z, allowlist="0123456789")
    right_raw = _ocr_text(right_z, allowlist="0123456789")

    # For center allow arabic letters only (plus maybe noise)
    center_raw = _ocr_text(center_z)  # keep full, then extract valid

    left_digits = _only_digits(left_raw)
    right_digits = _only_digits(right_raw)

    arabic_letter = _extract_valid_arabic(center_raw)

    # if OCR gave nothing useful, use shape fallback
    if not arabic_letter:
        arabic_letter = _detect_arabic_letter_shape(center_z)

    # FINAL FORMAT
    parts = []
    if left_digits:
        parts.append(left_digits)
    if arabic_letter:
        parts.append(arabic_letter)
    if right_digits:
        parts.append(right_digits)

    final = " ".join(parts) if parts else None

    if not return_debug:
        return final

    debug = {
        "gray": gray,
        "enhanced": th,
        "separators": {"s1": seps[0], "s2": seps[1]},
        "zones": {"left": left_z, "center": center_z, "right": right_z},
        "zone_raw": {"left": left_raw, "center": center_raw, "right": right_raw},
        "final": final,
    }
    return final, debug

# =====================================
# CLI TEST
# =====================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python read_plate_moroccan.py image.jpg")
        raise SystemExit(1)

    print("Detected plate:", read_moroccan_plate(sys.argv[1]))
