import os
import csv
import cv2
from detect_plates import safe_imread
from ocr_engine import load_ocr_model, segment_plate, predict_char
from difflib import SequenceMatcher

GT_CSV = "data/eval/ground_truth.csv"
PLATE_IMAGES_DIR = "outputs/plates"

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def eval_pipeline():
    model, idx_to_char = load_ocr_model()

    total = 0
    perfect = 0
    char_acc_sum = 0

    with open(GT_CSV, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            img_path = os.path.join(PLATE_IMAGES_DIR, row["image_name"])
            true_text = row["true_plate"]

            if not os.path.exists(img_path):
                continue

            img = safe_imread(img_path)
            chars = segment_plate(img)

            pred_text = ""
            for c in chars:
                pred_text += predict_char(model, idx_to_char, c)

            sim = similarity(pred_text, true_text)
            char_acc_sum += sim

            if pred_text == true_text:
                perfect += 1

            total += 1
            print(f"{row['image_name']} | GT={true_text} | PRED={pred_text} | SIM={round(sim*100,1)}%")

    print("\n================ FINAL PIPELINE KPIs ================")
    print(f"Total Plates Tested : {total}")
    print(f"Perfect Plates      : {perfect}")
    print(f"Plate Accuracy (%)  : {round((perfect/total)*100,2)}")
    print(f"Character Accuracy  : {round((char_acc_sum/total)*100,2)}")
    print("=====================================================")

if __name__ == "__main__":
    eval_pipeline()
