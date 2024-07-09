import streamlit as st
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import os
import requests

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Rock Paper Scissors Classification')

st.markdown("Welcome to the Rock Paper Scissors Classification using CNN for Dicoding Machine Learning Last Submission.")

def download_model(url, namefile):
    response = requests.get(url)
    with open(namefile, 'wb') as f:
        f.write(response.content)

def predict(image_name):
    classifier_model = "rps-dicoding.h5"
    model_url = 'https://github.com/FariskaRatna/Rock-paper-scissors-Dicoding-ML/releases/download/v1_rps/rps-dicoding.h5'

    if not os.path.exists(classifier_model):
        download_model(model_url, classifier_model)

    model = load_model(classifier_model)

    img = image.load_img(image_name, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    if classes[0, 0] != 0:
        result = "PAPER"
    elif classes[0, 1] != 0:
        result = "ROCK"
    else:
        result = "SCISSORS"

    results = f"The classification of the image is {result} with a confidence score of {100 * np.max(classes):.2f}%"

    return results

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    class_button = st.button("Classifier")
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption="Image has been uploaded", use_column_width=True, width=300)

    if class_button:
        if file_uploaded is None:
            st.write("Please upload an image")
        else:
            with st.spinner("Model working..."):
                predictions = predict(file_uploaded)
                time.sleep(1)
                st.success("Classified")
                st.write(predictions)

if __name__ == "__main__":
    main()
