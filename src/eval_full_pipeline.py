import os
import csv
from difflib import SequenceMatcher

from read_plate_english import read_english_plate
from read_plate_moroccan import read_moroccan_plate


# ================= PATHS =================
PLATES_DIR = "outputs/plates"
GT_CSV = "data/eval/ground_truth.csv"

USABLE_THRESHOLD = 0.70  # 70% characters correct


# ================= UTILS =================
def similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def safe_ocr(fn, img_path):
    try:
        out = fn(img_path)
    except:
        return "", 0.0

    if isinstance(out, tuple):
        return out[0].strip(), float(out[1])

    if isinstance(out, str):
        return out.strip(), 0.5

    return "", 0.0


def run_ocr(img_path):
    en_text, en_conf = safe_ocr(read_english_plate, img_path)
    ma_text, ma_conf = safe_ocr(read_moroccan_plate, img_path)

    if ma_conf >= en_conf and len(ma_text) >= len(en_text):
        return ma_text.upper(), ma_conf

    return en_text.upper(), en_conf


# ================= EVAL =================
def evaluate():
    print("\n🚀 RUNNING OCR EVALUATION\n")

    if not os.path.exists(GT_CSV):
        print("❌ Missing CSV")
        return

    if not os.path.isdir(PLATES_DIR):
        print("❌ Missing plates directory")
        return

    total = 0
    exact_match = 0
    usable = 0

    char_acc_sum = 0
    cer_sum = 0
    precision_sum = 0
    recall_sum = 0

    missing = 0

    with open(GT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            img_name = row["image_name"].strip()
            gt = row["true_plate"].strip().upper()

            img_path = os.path.join(PLATES_DIR, img_name)
            if not os.path.exists(img_path):
                missing += 1
                continue

            pred, _ = run_ocr(img_path)

            sim = similarity(pred, gt)
            char_acc_sum += sim
            cer_sum += (1 - sim)

            if pred == gt:
                exact_match += 1

            if sim >= USABLE_THRESHOLD:
                usable += 1

            # precision / recall (character-level)
            common = sum((c in gt) for c in pred)
            precision = common / len(pred) if pred else 0
            recall = common / len(gt) if gt else 0

            precision_sum += precision
            recall_sum += recall

            total += 1

    print("\n=============== OCR KPIs (FINAL) ================")
    print(f"Total samples               : {total}")
    print(f"Exact Plate Accuracy (%)    : {round((exact_match/total)*100,2) if total else 0}")
    print(f"Usable Plate Rate ≥70% (%)  : {round((usable/total)*100,2) if total else 0}")
    print(f"Avg Character Accuracy (%)  : {round((char_acc_sum/total)*100,2) if total else 0}")
    print(f"Avg CER (↓ better) (%)      : {round((cer_sum/total)*100,2) if total else 0}")
    print(f"Avg Precision (chars) (%)   : {round((precision_sum/total)*100,2) if total else 0}")
    print(f"Avg Recall (chars) (%)      : {round((recall_sum/total)*100,2) if total else 0}")
    print(f"Missing images              : {missing}")
    print("================================================\n")


# ================= RUN =================
if __name__ == "__main__":
    evaluate()
