from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder="resources", static_folder="dist")

@app.route('/')
def index():
    return render_template('views/main.html')

def preprocess_image(img):
  # Convert from BGR (OpenCV format) to RGB
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # Resize the image to match the model's input shape (256x256)
  img = cv2.resize(img, (256, 256))
  # Normalize pixel values to range [0, 1]
  img = img.astype('float32') / 255.0
  # Add an extra dimension for batch processing (even though we only have 1 image)
  return img

@app.route('/detect', methods=['POST'])
def detect():
    image_file = request.files.get('image')

    data_dir = 'app/Data'
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_data = train_datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )
    class_labels = list(train_data.class_indices.keys())

    if image_file:
        image_data = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = preprocess_image(image_data)

        model = load_model('app/car_select.keras')
        prediction = model.predict(np.expand_dims(image, axis=0))[0]

        predicted_class = np.argmax(prediction)
        return jsonify({'predicted_class': str(class_labels[predicted_class]), 'probability': str(prediction[predicted_class])})
    else:
        return "No image uploaded!", 400