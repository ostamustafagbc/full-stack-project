import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.utils import to_categorical  # for one-hot encoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # for data augmentation (optional)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'Data'
os.listdir(data_dir)

image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
  for image in os.listdir(os.path.join(data_dir, image_class)):
    image_path = os.path.join(data_dir, image_class, image)
    try:
      img = cv2.imread(image_path)
      tip = imghdr.what(image_path)
      if tip not in image_exts:
        os.remove(image_path)
    except Exception as e:
      print('Issue with image {}'.format(image_path))

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  # optional data augmentation
val_datagen = ImageDataGenerator(rescale=1./255)  # no data augmentation for validation

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,  # adjust batch size as needed
    class_mode='categorical'  # specify categorical labels
)

val_data = val_datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,  # adjust batch size as needed
    class_mode='categorical'  # specify categorical labels
)

car_manufacturers = len(train_data.class_indices)  # get number of classes from training data

model = Sequential()

# Step 1 - Convolution
# Use ReLU (rectifier function) to delete negative pixels to avoid non linearities
model.add(Conv2D(16, (3, 3), 1, input_shape=(256, 256, 3),
                  activation='relu'))
model.add(MaxPooling2D())

# Step 2 - Pooling
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

# Adding a second convolutional layer
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection

model.add(Dense(256, activation='relu'))
model.add(Dense(car_manufacturers, activation='softmax'))  # Adjust for number of classes

# Model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy', 'categorical_accuracy'])

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train_data, epochs=20,
                validation_data=val_data,
                callbacks=[tensorboard_callback])
model.save('car_select.keras')

# Evaluation (assuming one-hot encoded labels)
acc = CategoricalAccuracy()

for x, y in val_data:
  yhat = model.predict(x)
  print(y)
  print(x)
  print(yhat)
  break
  acc.update_state(y, yhat)

  print(f'Accuracy:{acc.result().numpy()}')