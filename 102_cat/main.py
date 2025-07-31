from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import json

app = Flask(__name__)



# MODEL = tf.keras.models.load_model("model/pre_trained_flower_classification_model_v1.h5")

# Load mappings
with open("dataset/cat_to_name.json", "r") as f:
    class_names = json.load(f)

with open("dataset/class_order.json", "r") as f:
    class_order = json.load(f)

with open("dataset/class_names.json", "r") as f:
    class_names_5cat = json.load(f)


def read_file_as_image(data):
    image = Image.open(BytesIO(data)).resize((256, 256))
    
    return image


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files or 'model_choice' not in request.form:
        return jsonify({'error': 'Missing file or model choice'}), 400

    file = request.files['file']
    model_choice = request.form['model_choice']

    print(f"Model: {model_choice}")

    try:
        model_path = f"model/{model_choice}"
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return jsonify({'error': f"Failed to load model: {str(e)}"}), 500

    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = model.predict(img_batch)

    if model_choice == "pre_trained_flower_classification_model_v1.h5":
        
        pred_index = np.argmax(predictions)
        class_id = class_order[pred_index]
        flower_name = class_names[class_id]
        confidence = np.max(predictions)
    else:
        flower_name = class_names_5cat[np.argmax(predictions)]
        confidence = np.max(predictions)

    # print(f"Predicted class: {flower_name}, Confidence: {confidence}, Class number: {class_id}")

    return jsonify({
        'predicted_class': flower_name,
        # 'Predicted_class_number': class_id,
        'confidence': float(confidence)
    })
     










# This way we can run the flask app without adding it to the environment variable:
if __name__ == "__main__":
    app.run(port=8001, debug=True)  # Set debug=True for development purposes