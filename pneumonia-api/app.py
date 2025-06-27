import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

model = load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Pneumonia Detection API is live!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_path = os.path.join(BASE_DIR, "temp.jpg")
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(120, 120))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

    os.remove(img_path)

    return jsonify({
        "prediction": label,
        "confidence": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)
