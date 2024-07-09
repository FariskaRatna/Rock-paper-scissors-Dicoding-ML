import streamlit as st
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import os

# Function to download the model if not present
def download_model(url, namefile):
    response = requests.get(url)
    with open(namefile, 'wb') as f:
        f.write(response.content)

# Function to classify the image
def classify_image(img, model):
    # Convert the image to a numpy array
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Predict the class
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    # Determine the class label
    if classes[0, 0] != 0:
        return 'PAPER'
    elif classes[0, 1] != 0:
        return 'ROCK'
    else:
        return 'SCISSORS'

# Function to load the model
def load_trained_model():
    classifier_model = "rps-dicoding.h5"
    model_url = 'https://github.com/FariskaRatna/Rock-paper-scissors-Dicoding-ML/releases/download/v1_rps/rps-dicoding.h5'

    if not os.path.exists(classifier_model):
        download_model(model_url, classifier_model)

    model = load_model(classifier_model)
    return model

# Load the model once at the start
model = load_trained_model()

# Streamlit interface
st.title('Rock Paper Scissors Classification')
st.markdown("Welcome to the Rock Paper Scissors Classification using CNN for Dicoding Machine Learning Last Submission.")

file_uploaded = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if file_uploaded is not None:
    # Display the uploaded image
    img = Image.open(file_uploaded)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Classify the uploaded image
    st.write("Classifying...")
    label = classify_image(img, model)
    st.write(f'The classification of the image is: **{label}**')
