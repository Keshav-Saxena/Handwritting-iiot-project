"""Image preprocessing (Report Ch. 7): grayscale -> polarity fix -> crop -> centre -> 28x28 -> normalise.
The SAME function is used for every prediction so training/prediction stay consistent."""
import os
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_BYTES = 10 * 1024 * 1024


def load_gray(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in ALLOWED:
        raise ValueError("Unsupported file type. Please upload a PNG or JPG file.")
    if os.path.getsize(path) > MAX_BYTES:
        raise ValueError("Image file is too large (limit 10 MB).")
    try:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError):
        raise ValueError("The image could not be read. Please upload a valid PNG or JPG file.")


def preprocess_array(gray):
    """gray: 2-D uint8 array. Returns (1,28,28,1) float32 in [0,1] and the 28x28 uint8 image."""
    h, w = gray.shape
    if max(h, w) > 800:
        s = 800 / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    # MNIST = light digit on dark background. If the border is light, invert.
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if np.median(border) > 127:
        gray = 255 - gray

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if int(gray.max()) - int(gray.min()) < 40:
        raise ValueError("Blank or very low-contrast image. Please write one clear digit.")

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        raise ValueError("No digit found in the image.")

    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    digit = np.where(mask[y1:y2, x1:x2] > 0, gray[y1:y2, x1:x2], 0).astype(np.uint8)

    # Scale longest side to 20 px (MNIST style), then centre by centre of mass in 28x28
    dh, dw = digit.shape
    s = 20.0 / max(dh, dw)
    digit = cv2.resize(digit, (max(1, round(dw * s)), max(1, round(dh * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y0 = (28 - digit.shape[0]) // 2
    x0 = (28 - digit.shape[1]) // 2
    canvas[y0:y0 + digit.shape[0], x0:x0 + digit.shape[1]] = digit
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]]), (28, 28))

    arr = (canvas.astype("float32") / 255.0).reshape(1, 28, 28, 1)
    return arr, canvas


def preprocess_image(path):
    return preprocess_array(load_gray(path))
