try:
    from tensorflow.keras.models import load_model
    print("TensorFlow imported successfully")
except Exception as e:
    print("TensorFlow import failed:", e)
    exit()

model_path = "models/best_cnn_model.h5"

try:
    model = load_model(model_path)
    print("Model loaded successfully")
    model.summary()
except Exception as e:
    print("Failed to load model:", e)
