from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
import librosa

app = Flask(__name__)

# Load your models
image_model = load_model('models/asd_image_model.keras')  # Make sure the file name matches
audio_model = load_model('models/asd_audio_model.keras')  # Make sure the file name matches

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return render_template('result.html', prediction="❌ No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', prediction="❌ No file selected!")

    try:
        # Correct way to load image from stream
        img = load_img(file.stream, target_size=(128, 128))  # adjust target size to your model
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # model expects batch dimension

        prediction = image_model.predict(img)
        result = np.argmax(prediction)

        label = "Autistic" if result == 0 else "Non-Autistic"

        return render_template('result.html', prediction=label)
    except Exception as e:
        print(f"Image Prediction error: {e}")
        return render_template('result.html', prediction="❌ Prediction failed. Please try again!")

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return render_template('result_audio.html', prediction="❌ No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('result_audio.html', prediction="❌ No file selected!")

    try:
        # Correct way to load audio from stream
        y, sr = librosa.load(file.stream, sr=None)

        # Example feature extraction (modify according to your model's input)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        mfccs = np.expand_dims(mfccs, axis=0)  # model expects batch dimension

        prediction = audio_model.predict(mfccs)
        result = np.argmax(prediction)

        label = "Autistic" if result == 0 else "Non-Autistic"

        return render_template('result_audio.html', prediction=label)
    except Exception as e:
        print(f"Audio Prediction error: {e}")
        return render_template('result_audio.html', prediction="❌ Audio prediction failed. Please try again!")

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
