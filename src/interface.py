import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

from read_plate_english import read_english_plate
from read_plate_moroccan import read_moroccan_plate

# ============================
# LOAD YOLO MODEL
# ============================
MODEL_PATH = "models/yolo/lp_detector_v11_best.pt"
detector = YOLO(MODEL_PATH)

# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="ALPR System", layout="centered")
st.title("ALPR – License Plate Recognition System")

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

plate_type = st.selectbox(
    "Select Plate Type",
    ["English / International", "Moroccan"]
)

# ============================
# IMAGE PROCESSING
# ============================
if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Full Input Image")

    if st.button("Detect & Read Plate"):

        with st.spinner("Detecting plate with YOLO..."):
            results = detector(img, conf=0.40)

        if len(results) == 0 or results[0].boxes is None:
            st.error(" No plate detected by YOLO.")
        else:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            # ✅ SELECT LARGEST BOX
            largest_box = None
            largest_area = 0

            for b in boxes:
                x1_, y1_, x2_, y2_ = map(int, b[:4])
                area = (x2_ - x1_) * (y2_ - y1_)
                if area > largest_area:
                    largest_area = area
                    largest_box = (x1_, y1_, x2_, y2_)

            if largest_box is None:
                st.error("YOLO detected a box but it is invalid.")
            else:
                x1, y1, x2, y2 = largest_box

                # ============================
                # ✅ STEP 1 — RAW YOLO BOX
                # ============================
                raw_box_img = img.copy()
                cv2.rectangle(raw_box_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                st.image(raw_box_img, channels="BGR", caption="Step 1: Raw YOLO Detection")

                # ============================
                # ✅ STEP 2 — EXPANDED BOX (MOROCCAN FIX)
                # ============================
                h, w, _ = img.shape
                plate_width = x2 - x1
                plate_height = y2 - y1

                expand_x = int(plate_width * 0.45)
                expand_y = int(plate_height * 0.30)

                x1e = max(0, x1 - expand_x)
                x2e = min(w, x2 + expand_x)
                y1e = max(0, y1 - expand_y)
                y2e = min(h, y2 + expand_y)

                expanded_img = img.copy()
                cv2.rectangle(expanded_img, (x1e, y1e), (x2e, y2e), (255, 0, 0), 2)
                st.image(expanded_img, channels="BGR", caption="Step 2: Expanded Detection")

                # ============================
                # ✅ STEP 3 — FINAL CROP SENT TO OCR
                # ============================
                crop = img[y1e:y2e, x1e:x2e]

                if crop.size == 0:
                    st.error("Cropped plate is empty.")
                else:
                    st.image(crop, channels="BGR", caption=" Step 3: Final OCR Crop")

                    # ============================
                    # SAVE TEMP IMAGE
                    # ============================
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        cv2.imwrite(tmp.name, crop)
                        crop_path = tmp.name

                    # ============================
                    # OCR
                    # ============================
                    with st.spinner("Reading plate with OCR..."):
                        if plate_type == "English / International":
                            result = read_english_plate(crop_path)
                        else:
                            result = read_moroccan_plate(crop_path)

                    os.remove(crop_path)

                    # ============================
                    # DISPLAY RESULT
                    # ============================
                    if result:
                        st.success(f"Detected Plate: {result}")
                    else:
                        st.error("OCR failed to read the plate.")
