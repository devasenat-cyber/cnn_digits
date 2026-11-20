"""
Simple Flask API for the trained MNIST CNN.

Expected model files:
    model/digit_classifier.keras
    model/class_names.json

Run:
    python app.py

Then open:
    http://127.0.0.1:5000/
"""

import io
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError


# ---------------------------------------------------------
# 1. Create the Flask application
# ---------------------------------------------------------

app = Flask(__name__)

# Allow uploads up to 5 MB.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Find the model folder beside this app.py file.
BASE_FOLDER = Path(__file__).resolve().parent
MODEL_FOLDER = BASE_FOLDER / "model"

MODEL_PATH = MODEL_FOLDER / "digit_classifier.keras"
CLASS_NAMES_PATH = MODEL_FOLDER / "class_names.json"


# ---------------------------------------------------------
# 2. Load the trained CNN only once when the app starts
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "\nTrained model was not found!\n"
        f"Copy digit_classifier.keras into this folder:\n{MODEL_FOLDER}\n"
    )

print("Loading the trained CNN...")
model = tf.keras.models.load_model(MODEL_PATH)
print("CNN model loaded successfully!")

# Load the class names if the JSON file is available.
# If it is missing, use the output indexes: 0, 1, 2, ...
if CLASS_NAMES_PATH.exists():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)
else:
    class_names = [str(i) for i in range(int(model.output_shape[-1]))]

# Check that the labels match the model's output layer.
if len(class_names) != int(model.output_shape[-1]):
    raise ValueError(
        "The number of class names does not match the model output layer."
    )


# ---------------------------------------------------------
# 3. Prepare the uploaded image
# ---------------------------------------------------------

def prepare_image(image_file):
    """
    Convert uploaded image into the format used during MNIST training:
    28 x 28 pixels, grayscale, values between 0 and 1.
    """

    image = Image.open(io.BytesIO(image_file))
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    image_array = np.array(image, dtype=np.float32) / 255.0

    # CNN expects: (batch, height, width, channels)
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


# ---------------------------------------------------------
# 4. Home API
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the MNIST Digit Classifier API!",
        "model_loaded": True,
        "available_apis": {
            "home": "GET /",
            "health": "GET /health",
            "predict": "POST /predict"
        },
        "how_to_predict": (
            "Send a POST request to /predict with an image "
            "using the form-data key 'file'."
        )
    })


# ---------------------------------------------------------
# 5. Health-check API
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None
    })


# ---------------------------------------------------------
# 6. Prediction API
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether the request contains a file.
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded. Use the form-data key 'file'."
        }), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Please select an image."
        }), 400

    try:
        # Read the uploaded image.
        image_bytes = uploaded_file.read()

        # Convert it to the same input format used during training.
        input_image = prepare_image(image_bytes)

        # Ask the trained CNN to predict.
        start_time = time.perf_counter()
        predictions = model.predict(input_image, verbose=0)[0]
        inference_time = (time.perf_counter() - start_time) * 1000

        # Get the class with the highest probability.
        predicted_index = int(np.argmax(predictions))
        predicted_name = str(class_names[predicted_index])
        confidence = float(predictions[predicted_index])

        # Return the three most likely classes.
        top_three = np.argsort(predictions)[::-1][:3]

        return jsonify({
            "status": "success",
            "filename": uploaded_file.filename,
            "predicted_class": predicted_name,
            "class_id": predicted_index,
            "confidence": round(confidence, 4),
            "top_3_predictions": [
                {
                    "class": str(class_names[int(index)]),
                    "probability": round(float(predictions[index]), 4)
                }
                for index in top_three
            ],
            "inference_time_ms": round(float(inference_time), 2)
        })

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return jsonify({
            "status": "error",
            "message": f"Could not process the uploaded image: {str(error)}"
        }), 400


# ---------------------------------------------------------
# 7. Start the Flask server
# ---------------------------------------------------------

if __name__ == "__main__":
    print("\nFlask API is starting...")
    print("Open http://127.0.0.1:5000/")
    print("Health check: http://127.0.0.1:5000/health\n")

    app.run(host="127.0.0.1", port=5000, debug=True)
