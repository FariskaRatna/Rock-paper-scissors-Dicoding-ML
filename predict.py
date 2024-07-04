import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from keras.preprocessing import image

def prediction(image_path):
    base_model='model/rps-dicoding.h5'
    IMAGE_SHAPE = (224, 224, 3)
    model = load_model(base_model)

    test_image = image.load_img(image_path, target_size=(224,224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    test_image = np.vstack([test_image])
    classes = model.predict(test_image, batch_size=10)

    paper_confidence = classes[0,0]
    rock_confidence = classes[0,1]
    scissor_confidence = classes[0,2]

    if paper_confidence != 0:
        print(f'PAPER with confidence {paper_confidence:.2f}')
    elif classes[0,1] != 0:
        print(f'ROCK with confidence {rock_confidence:.2f}')
    else:
        print(f'SCISSORS with confidence {scissor_confidence:.2f}')


image_path = "thumb.jpg"
predictions = prediction(image_path)
    