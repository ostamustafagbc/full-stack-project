import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from urllib.request import urlopen


# Load the saved model
model = load_model('car_select.keras')

def download_image(url):
  response = urlopen(url)
  return np.asarray(bytearray(response.read()), dtype="uint8")

def preprocess_image(img):
  # Convert from BGR (OpenCV format) to RGB
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # Resize the image to match the model's input shape (256x256)
  img = cv2.resize(img, (256, 256))
  # Normalize pixel values to range [0, 1]
  img = img.astype('float32') / 255.0
  # Add an extra dimension for batch processing (even though we only have 1 image)
  img = np.expand_dims(img, axis=0)
  return img

# Load the saved model
model = load_model('car_select.keras')

# Get the path to your new image
image_url = 'https://www.usatoday.com/gcdn/-mm-/37fe7faa7fed7edcc51e71fe40f11cfe37c3daaa/c=131-0-2174-1536/local/-/media/2018/01/19/USATODAY/USATODAY/636519786314510593-18-FUSI-SE-34FrntPassStaticRooftop-mj.jpg'  # Replace with your actual path

# Download the image
img = download_image(image_url)

# Preprocess the image
preprocessed_image = preprocess_image(img)

# Make prediction using the model
prediction = model.predict(preprocessed_image)

# Get the class labels (car brand names) from the training data
data_dir = 'Data'
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,  # adjust batch size as needed
    class_mode='categorical'  # specify categorical labels
)
class_labels = list(train_data.class_indices.keys())  # Inverted dictionary from index to class name

# Find the predicted class index with the highest probability
predicted_class_idx = np.argmax(prediction[0])

# Get the predicted car brand name using the class index and class labels

predicted_car_brand = class_labels[predicted_class_idx]

# Print the predicted car brand
print(f"Predicted car brand: {predicted_car_brand}")