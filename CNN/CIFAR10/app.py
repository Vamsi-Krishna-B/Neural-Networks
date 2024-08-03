from flask import Flask, request, redirect, url_for,render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = load_model('keras_cifar10_trainedmodel.h5')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

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
    img = img.resize((32, 32))  # CIFAR-10 image size
    img = np.array(img) / 255.0  # Normalize pixel values

    # Predict the class of the image
    predictions = model.predict(np.expand_dims(img, axis=0))
    predicted_class = class_names[np.argmax(predictions)]
    print(predicted_class)
    return render_template('result.html', predicted_class=predicted_class)

    # Redirect to the result page with the predicted class as a query parameter
    

if __name__ == '__main__':
    app.run(debug=True)
