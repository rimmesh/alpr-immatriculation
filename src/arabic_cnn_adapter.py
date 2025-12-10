import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load your exact model
arabic_model = load_model("models/best_cnn_model.h5")

# ⚠️ MUST MATCH YOUR TRAINING LABEL ORDER
ARABIC_LABELS = [
    'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر',
    'ز','س','ش','ص','ض','ط','ظ','ع','غ','ف',
    'ق','ك','ل','م','ن','ه','و','ي'
]

# ✅ THIS FUNCTION ADAPTS IMAGE TO YOUR CNN
def predict_arabic_letter_from_crop(letter_crop):

    # ✅ Convert to grayscale
    gray = cv2.cvtColor(letter_crop, cv2.COLOR_BGR2GRAY)

    # ✅ Resize EXACTLY to what Dense expects:
    # Your error showed: model expects 4608 features
    # 4608 = 48 × 96
    gray = cv2.resize(gray, (96, 48))

    # ✅ Normalize
    gray = gray.astype("float32") / 255.0

    # ✅ Flatten (THIS was missing before)
    gray = gray.reshape(1, -1)   # (1, 4608)

    # ✅ Predict
    pred = arabic_model.predict(gray, verbose=0)
    idx = np.argmax(pred)

    return ARABIC_LABELS[idx]
