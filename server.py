from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import librosa
import os

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
image_model = load_model('model/asd_image_model.keras')
audio_model = load_model('model/asd_audio_model.keras')

# Class labels
class_labels = ['Non-Autistic', 'Autistic']

# ==== Helper Functions ====
def load_audio(audio_path, max_length=100):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.T
    if mfcc.shape[0] > max_length:
        mfcc = mfcc[:max_length, :]
    else:
        pad_width = max_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    return mfcc

# ==== Routes ====

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename != '':
            img = load_img(image_file, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            image_prediction = image_model.predict(img_array)[0][0]
            predicted_class = 1 if image_prediction > 0.5 else 0
            confidence = image_prediction if predicted_class == 1 else 1 - image_prediction
            label = class_labels[predicted_class]

            return f"Image Prediction: {label} ({confidence * 100:.2f}% confidence)"

    if 'audio' in request.files:
        audio_file = request.files['audio']
        if audio_file.filename != '':
            audio_path = "temp_audio.wav"
            audio_file.save(audio_path)

            audio_data = np.expand_dims(load_audio(audio_path), axis=0)

            audio_prediction = audio_model.predict(audio_data)[0][0]
            predicted_class = 1 if audio_prediction > 0.5 else 0
            confidence = audio_prediction if predicted_class == 1 else 1 - audio_prediction
            label = class_labels[predicted_class]

            os.remove(audio_path)  # cleanup temp file
            return f"Audio Prediction: {label} ({confidence * 100:.2f}% confidence)"

    return "No valid file uploaded."

# ==== Run App ====
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
