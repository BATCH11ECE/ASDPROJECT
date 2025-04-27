import os
import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
from keras.utils import load_img, img_to_array
import librosa

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
image_model = load_model(os.path.join(BASE_DIR, 'model', 'asd_image_model.keras'))
audio_model = load_model(os.path.join(BASE_DIR, 'model', 'asd_audio_model.keras'))

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' in request.files and request.files['image'].filename != '':
            image_file = request.files['image']
            img = load_img(image_file.stream, target_size=(128, 128))  # Correct way
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            prediction = image_model.predict(img_array)
            result = 'ASD Detected' if prediction[0][0] > 0.5 else 'No ASD Detected'
            return render_template('index.html', prediction=result)

        elif 'audio' in request.files and request.files['audio'].filename != '':
            audio_file = request.files['audio']
            audio_path = os.path.join(BASE_DIR, 'temp_audio.wav')
            audio_file.save(audio_path)

            y, sr = librosa.load(audio_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc = np.mean(mfcc.T, axis=0)
            mfcc = np.expand_dims(mfcc, axis=0)

            prediction = audio_model.predict(mfcc)
            result = 'ASD Detected' if prediction[0][0] > 0.5 else 'No ASD Detected'
            os.remove(audio_path)
            return render_template('index.html', prediction=result)

        else:
            return render_template('index.html', prediction='❌ No file uploaded.')

    except Exception as e:
        print('Error during prediction:', str(e))
        return render_template('index.html', prediction='❌ Prediction failed. Please try again!')

# Run server
if __name__ == '__main__':
    app.run(debug=True)
