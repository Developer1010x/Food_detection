import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO
import tensorflow_hub as hub
from bs4 import BeautifulSoup

# Define the path to the user-uploaded image
user_image_path = '/home/hacker69i/python code/food-101/images/spaghetti_bolognese/62690.jpg'

# Define the path to your Food-101 dataset directory
dataset_dir = '/home/hacker69i/python code/food-101'

# Define the path to the images directory
images_dir = os.path.join(dataset_dir, 'images')
model_url = "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4"
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(299, 299, 3)),
    hub.KerasLayer(model_url, output_shape=[1001])
])

def classify_user_image(image_path):
    img = Image.open(image_path)
    img = img.resize((299, 299))
    img = np.array(img) / 255.0
    predictions = model.predict(np.expand_dims(img, axis=0))

    class_labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    class_labels = requests.get(class_labels_url).text.split('\n')

    top_class = class_labels[np.argmax(predictions)]

    return top_class



# Classify the user-entered image
food_prediction = classify_user_image(user_image_path)

print(f"The detected food is: {food_prediction}")
