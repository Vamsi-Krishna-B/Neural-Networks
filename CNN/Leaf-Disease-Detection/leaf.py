from flask import Flask, request, redirect, url_for,render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = load_model('Trained_leaf_Detection.keras')
class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image was properly uploaded to our endpoint
    if 'image' not in request.files:
        return redirect(url_for('index'))

    # Read the image
    image_file = request.files['image']
    img = Image.open(image_file)

    # Preprocess the image
    img = img.resize((64, 64))  # CIFAR-10 image size
    img = np.array(img) / 255.0  # Normalize pixel values

    # Predict the class of the image
    predictions = model.predict(np.expand_dims(img, axis=0))
    predicted_class = class_names[np.argmax(predictions)]
    print(predicted_class)
    return render_template('result.html', predicted_class=predicted_class)

    # Redirect to the result page with the predicted class as a query parameter
    

if __name__ == '__main__':
    app.run(debug=True)
