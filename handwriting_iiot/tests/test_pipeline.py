"""Tests for preprocessing + MQTT message (Report Ch.12 unit tests / TC-03..TC-06, TC-08). No TensorFlow needed."""
import sys, os, json, tempfile
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from preprocess import preprocess_image
from mqtt_publish import build_message, publish_result

SAMP = os.path.join(os.path.dirname(__file__), "..", "samples"); os.makedirs(SAMP, exist_ok=True)
rng = np.random.default_rng(0)

def make(d, size=400, tilt=0, noise=0, thick=18, dark_ink=True, scale=5, off=(0, 0)):
    img = np.full((size, size), 235 if dark_ink else 20, np.uint8)
    cv2.putText(img, str(d), (size//2-60+off[0], size//2+70+off[1]), cv2.FONT_HERSHEY_SIMPLEX, scale,
                30 if dark_ink else 240, thick, cv2.LINE_AA)
    if tilt: img = cv2.warpAffine(img, cv2.getRotationMatrix2D((size/2, size/2), tilt, 1), (size, size),
                                  borderValue=235 if dark_ink else 20)
    if noise: img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img

results = []
def check(name, cond, detail=""):
    results.append(cond); print(("PASS " if cond else "FAIL ") + name, detail)

cases = {"TC-01 digit 3 clear": make(3), "TC-02 digit 8 clear": make(8),
         "TC-03 digit 5 tilted 12deg": make(5, tilt=12), "noisy phone-style 7": make(7, noise=12),
         "light-on-dark 2 (MNIST polarity)": make(2, dark_ink=False), "off-centre small 4": make(4, scale=3, off=(90, -60))}
tiles = []
for name, img in cases.items():
    p = os.path.join(SAMP, name.split()[0].replace(":", "") + "_" + name.replace(" ", "_").replace("(", "").replace(")", "") + ".png")
    cv2.imwrite(p, img)
    arr, prev = preprocess_image(p)
    ok = arr.shape == (1, 28, 28, 1) and arr.dtype == np.float32 and 0 <= arr.min() and arr.max() <= 1.0 and arr.max() > 0.5
    ys, xs = np.nonzero(prev > 40); cy, cx = ys.mean(), xs.mean()
    check(name, ok and abs(cx-14) < 3 and abs(cy-14) < 3, f"shape={arr.shape} range=[{arr.min():.2f},{arr.max():.2f}] centre=({cx:.1f},{cy:.1f}) corner_mean={prev[:3,:3].mean():.1f}")
    tiles.append(cv2.resize(prev, (140, 140), interpolation=cv2.INTER_NEAREST))
cv2.imwrite(os.path.join(os.path.dirname(__file__), "..", "results", "preprocessing_demo.png"), np.hstack(tiles))

def expect_error(name, path, frag):
    try: preprocess_image(path); check(name, False, "no error raised")
    except ValueError as e: check(name, frag.lower() in str(e).lower(), f'-> "{e}"')

tmp = tempfile.mkdtemp()
cv2.imwrite(f"{tmp}/blank.png", np.full((300, 300), 240, np.uint8)); expect_error("TC-04 blank image", f"{tmp}/blank.png", "blank")
open(f"{tmp}/x.txt", "w").write("hi"); expect_error("TC-05 unsupported file", f"{tmp}/x.txt", "unsupported")
open(f"{tmp}/bad.png", "wb").write(b"not an image"); expect_error("damaged image", f"{tmp}/bad.png", "could not be read")
big = make(6, size=3000, scale=40, thick=120); cv2.imwrite(f"{tmp}/big.png", big)
arr, _ = preprocess_image(f"{tmp}/big.png"); check("TC-08 large 3000x3000 image resized safely", arr.shape == (1, 28, 28, 1))

m = build_message(5, 0.9712); json.loads(json.dumps(m))
check("MQTT message valid JSON with required keys", set(m) == {"device_id","predicted_digit","confidence","timestamp","model_version"}, json.dumps(m))
os.environ["MQTT_BROKER"] = "127.0.0.1"
msg, ok, err = publish_result(5, 0.97)
check("TC-06 broker offline: no crash, delivery warning returned", ok is False and err is not None, f"err={err}")
print(f"\n{sum(results)}/{len(results)} tests passed"); sys.exit(0 if all(results) else 1)
