import os, numpy as np
from preprocess import preprocess_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = f"{ROOT}/models/digit_cnn.keras"
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model file missing. Run: python app/train_model.py")
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict_digit(image_path):
    arr, preview = preprocess_image(image_path)
    scores = get_model().predict(arr, verbose=0)[0]
    digit = int(np.argmax(scores))
    return digit, float(scores[digit]), preview


if __name__ == "__main__":
    import sys
    d, c, _ = predict_digit(sys.argv[1] if len(sys.argv) > 1 else "sample_digit.png")
    print("Predicted digit:", d); print("Confidence:", round(c * 100, 2), "%")
