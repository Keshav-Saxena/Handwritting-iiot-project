# IIoT-Based Handwritten Digit Recognition using CNN

## Setup
    python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## 1. Train (downloads MNIST; saves model, graphs, confusion matrix, metrics)
    python app/train_model.py          # optional: python app/train_model.py 10
Outputs go to `models/digit_cnn.keras` and `results/` (metrics.json, training_graphs.png,
confusion_matrix.png, classification_report.txt). Copy these numbers into Report Ch. 13.

## 2. Run the IIoT demo (3 terminals)
    mosquitto                          # MQTT broker (install Mosquitto first)
    python app/dashboard.py            # http://localhost:5001
    python app/app.py                  # http://localhost:5000  (upload a digit image)

Test without the web page:  `python app/predict.py samples/digit_3.png`
Test preprocessing (no TensorFlow needed): `python tests/test_pipeline.py`
Env vars: MQTT_BROKER, MQTT_PORT, MQTT_TOPIC (default lab/digit/result).
