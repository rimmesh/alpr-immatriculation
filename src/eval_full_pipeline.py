import os
import csv
from difflib import SequenceMatcher

# --------------------------------------------------
# PATHS (MATCH YOUR PROJECT)
# --------------------------------------------------
PLATES_DIR = "outputs/plates"
GT_CSV = "data/moroccan/ground_truth.csv"

# --------------------------------------------------
# IMPORT OCR
# --------------------------------------------------
from read_plate_english import read_english_plate
from read_plate_moroccan import read_moroccan_plate

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def safe_ocr_call(fn, img_path):
    """
    Forces OCR output to (text, confidence)
    """
    try:
        out = fn(img_path)
    except Exception as e:
        print(f"OCR crash on {img_path}: {e}")
        return "", 0.0

    # Case 1: already (text, conf)
    if isinstance(out, tuple) and len(out) == 2:
        return str(out[0]).strip(), float(out[1])

    # Case 2: text only
    if isinstance(out, str):
        return out.strip(), 0.5

    # Case 3: list (EasyOCR raw output)
    if isinstance(out, list) and len(out) > 0:
        try:
            best = max(out, key=lambda x: x[2])
            return best[1].replace(" ", "").upper(), float(best[2])
        except:
            return "", 0.0

    return "", 0.0

# --------------------------------------------------
# OCR DISPATCHER
# --------------------------------------------------
def run_ocr(img_path):
    en_text, en_conf = safe_ocr_call(read_english_plate, img_path)
    ma_text, ma_conf = safe_ocr_call(read_moroccan_plate, img_path)

    # Prefer Moroccan if confidence is close or higher
    if ma_conf >= en_conf and len(ma_text) >= len(en_text):
        return ma_text, ma_conf

    return en_text, en_conf

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
def evaluate():
    print("\n🚀 STARTING ALPR OCR EVALUATION\n")

    if not os.path.exists(GT_CSV):
        print(f"❌ Missing CSV: {GT_CSV}")
        return

    if not os.path.isdir(PLATES_DIR):
        print(f"❌ Missing plates folder: {PLATES_DIR}")
        return

    total = 0
    perfect = 0
    char_acc_sum = 0
    missing = 0

    with open(GT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not {"image_name", "true_plate"}.issubset(reader.fieldnames):
            print("❌ CSV must contain: image_name,true_plate")
            print("Found:", reader.fieldnames)
            return

        for row in reader:
            img_name = row["image_name"].strip()
            gt = row["true_plate"].strip().upper()

            img_path = os.path.join(PLATES_DIR, img_name)
            if not os.path.exists(img_path):
                missing += 1
                continue

            pred, conf = run_ocr(img_path)
            pred = pred.upper()

            sim = similarity(pred, gt)
            char_acc_sum += sim

            if pred == gt:
                perfect += 1

            total += 1
            print(
                f"{img_name} | GT={gt} | "
                f"PRED={pred} | CONF={round(conf,2)} | SIM={round(sim*100,1)}%"
            )

    print("\n================ FINAL OCR KPIs ================")
    print(f"Total plates tested : {total}")
    print(f"Perfect plates      : {perfect}")
    print(f"Plate accuracy (%)  : {round((perfect/total)*100,2) if total else 0}")
    print(f"Character accuracy  : {round((char_acc_sum/total)*100,2) if total else 0}")
    print(f"Missing images      : {missing}")
    print("================================================\n")

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    evaluate()
