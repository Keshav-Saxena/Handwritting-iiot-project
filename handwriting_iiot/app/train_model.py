"""Train the CNN from Report Ch. 8 / Appendix A on MNIST and record real metrics."""
import json, time, os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/models", exist_ok=True)
os.makedirs(f"{ROOT}/results", exist_ok=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test = (x_test.astype("float32") / 255.0)[..., None]

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

t0 = time.time()
hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=64, validation_split=0.1,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
train_time = time.time() - t0

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)   # evaluated once on held-out test set
y_pred = model.predict(x_test, verbose=0).argmax(1)

x1 = x_test[:1]; model.predict(x1, verbose=0)
t = time.time()
for _ in range(100): model.predict(x1, verbose=0)
inf_ms = (time.time() - t) / 100 * 1000

rep = classification_report(y_test, y_pred, digits=4, output_dict=True)
metrics = {"test_accuracy": float(test_acc), "test_loss": float(test_loss),
           "macro_precision": rep["macro avg"]["precision"], "macro_recall": rep["macro avg"]["recall"],
           "macro_f1": rep["macro avg"]["f1-score"], "training_time_s": round(train_time, 1),
           "avg_inference_ms": round(inf_ms, 2), "epochs_run": len(hist.history["loss"]),
           "tensorflow": tf.__version__}
json.dump(metrics, open(f"{ROOT}/results/metrics.json", "w"), indent=2)
open(f"{ROOT}/results/classification_report.txt", "w").write(classification_report(y_test, y_pred, digits=4))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist.history["accuracy"], label="train"); ax[0].plot(hist.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(hist.history["loss"], label="train"); ax[1].plot(hist.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{ROOT}/results/training_graphs.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion matrix (MNIST test set)"); plt.tight_layout()
plt.savefig(f"{ROOT}/results/confusion_matrix.png", dpi=150); plt.close()

model.save(f"{ROOT}/models/digit_cnn.keras")
print(json.dumps(metrics, indent=2))
