import os
import numpy as np
from flask import Flask, request, jsonify
from keras.utils import img_to_array, load_img
# from flask_ngrok import run_with_ngrok
from tensorflow import keras
# from keras import preprocess_input
import cv2
import PIL.Image



UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (128, 128)

app = Flask(__name__)
# run_with_ngrok(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the Keras model
model = keras.models.load_model('skin_classification.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(file):
    img = preprocess_input(file)
    prediction = model.predict(img)

    prediction = np.argmax(prediction, axis=1)
    print(prediction)
    if prediction == 0:
        return 'Acne'
    elif prediction == 1:
        return 'Eczema'
    elif prediction == 2:
        return 'Melanoma'
    elif prediction == 3:
        return 'Psoriasis'
    elif prediction == 4:
        return 'Onycholysis'

def preprocess_input(image):
    image = np.array(PIL.Image.open(image))
    resized_image = cv2.resize(image, dsize=(128, 128))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    reshaped_image = grayscale_image.reshape((128, 128, 3))
    reshaped_image = reshaped_image.reshape((1, 128, 128, 3))
    return reshaped_image
    
@app.route('/classify', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    info = predict(filepath)

    return jsonify({'code': '0', 'result': info}), 200
