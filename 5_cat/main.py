from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import json

app = Flask(__name__)



MODEL = tf.keras.models.load_model("models/flower_classification_5cat_model_v4.h5")

# Load mappings
with open("dataset/class_names.json", "r") as f:
    class_names = json.load(f)




def read_file_as_image(data):
    image = Image.open(BytesIO(data)).resize((256, 256))
    return image


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # pre_class_formatted = predicted_class.replace("_", " ").title()

    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    })     










# This way we can run the flask app without adding it to the environment variable:
if __name__ == "__main__":
    app.run(port=8001, debug=True)  # Set debug=True for development purposes