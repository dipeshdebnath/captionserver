import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 🔧 Config
MODEL_PATH = 'model/caption_model.h5'
TOKENIZER_PATH = 'model/tokenizer.pkl'
MAX_LENGTH = 22  # Make sure this matches your training script
IMAGE_SIZE = (224, 224)

# 🚀 Initialize
app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = pickle.load(open(TOKENIZER_PATH, 'rb'))
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# 🖼️ Feature extractor
def extract_features(image_path):
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return base_model.predict(image)

# 🔁 Temperature-based sampling
def sample_prediction(preds, temperature=0.7):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds[0]), p=preds[0])

# 🧠 Caption generator
def generate_caption(photo_features):
    in_text = 'startseq'
    generated_words = []

    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        preds = model.predict([photo_features, sequence], verbose=0)
        yhat = sample_prediction(preds, temperature=0.7)
        word = tokenizer.index_word.get(yhat)

        if word is None or word == 'endseq':
            break
        if generated_words[-5:].count(word) >= 3:
            break  # avoid excessive repetition
        generated_words.append(word)
        in_text += ' ' + word

    return ' '.join(generated_words)

# 📸 API endpoint
@app.route('/caption', methods=['POST'])
def caption_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    image_path = 'temp.jpg'
    image_file.save(image_path)

    try:
        features = extract_features(image_path)
        caption = generate_caption(features)
        os.remove(image_path)
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🏁 Run
if __name__ == '__main__':
    print("🚀 Flask server is starting... Visit http://localhost:5000/caption to test the API.")
    app.run(debug=True)
