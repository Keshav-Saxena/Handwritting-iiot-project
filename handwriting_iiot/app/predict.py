import os, numpy as np
from preprocess import preprocess_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = f"{ROOT}/models/digit_cnn.tflite"
_model = None


class _TFLiteModel:
    """Keras jaisa .predict() aur .input_shape deta hai, par TFLite se chalta hai."""

    def __init__(self, path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        self.interp = Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.input_shape = tuple(int(s) for s in self.inp["shape"])

    def predict(self, arr, verbose=0):
        x = np.asarray(arr, dtype=self.inp["dtype"])
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model file missing: digit_cnn.tflite (models folder mein upload karo)")
        _model = _TFLiteModel(MODEL_PATH)
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
