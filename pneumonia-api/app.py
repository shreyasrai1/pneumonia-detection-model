from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the model saved in the new Keras format
model = load_model("model.keras")

def preprocess_image(image, target_size=(120, 120)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files["image"]
    image = Image.open(image_file)
    processed = preprocess_image(image)

    prediction = model.predict(processed)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

    return jsonify({
        "prediction": label,
        "confidence": float(prediction)
    })

@app.route("/", methods=["GET"])
def index():
    return "✅ Pneumonia Detection API is running."

if __name__ == "__main__":
    app.run(debug=True)
