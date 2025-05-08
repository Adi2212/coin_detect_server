from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
import tempfile

# === Flask Setup ===
app = Flask(__name__)

# === Configurations ===
MODEL_PATH = 'my_model.pt'     # Path to YOLO model
MIN_THRESH = 0.50              # Confidence threshold

# Coin values in Indian Rupees
COIN_VALUES = {
    'one': 1.00,
    'two': 2.00,
    'five': 5.00,
    'ten': 10.00,
    'twenty': 20.00
}

# Load model at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("YOLO model file not found.")
model = YOLO(MODEL_PATH)
labels = model.names

@app.route('/')
def home():
    return "Welcome to the Coin Detection API!"

@app.route('/detect-coins', methods=['POST'])
def detect_coins():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded.'}), 400

        file = request.files['image']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            image = cv2.imread(temp.name)

        if image is None:
            return jsonify({'error': 'Invalid image.'}), 400

        results = model.predict(image, conf=MIN_THRESH, verbose=False)[0]
        detections = results.boxes

        detected_coins = []
        total_value = 0.0

        for box in detections:
            class_id = int(box.cls.item())
            class_name = labels[class_id]
            conf = box.conf.item()

            if conf > MIN_THRESH and class_name in COIN_VALUES:
                detected_coins.append(class_name)
                total_value += COIN_VALUES[class_name]

        return jsonify({
            'coins_detected': detected_coins,
            'count': len(detected_coins),
            'total_value_inr': f"{total_value:.2f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run App (optional) ===
# if __name__ == '__main__':
#     app.run(debug=True)
