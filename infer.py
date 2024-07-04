import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model


def classify_image(image_path, model):
    # Load the image
    img = image.load_img(image_path, target_size=(224, 224))
    
    # Display the image
    imgplot = plt.imshow(img)
    # plt.show()

    # Convert the image to a numpy array
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Predict the class
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    # Print the result
    if classes[0,0] != 0:
        print('PAPER')
    elif classes[0,1] != 0:
        print('ROCK')
    else:
        print('SCISSORS')

# Example usage
# Make sure to load your trained model here
model = load_model("model/rps-dicoding.h5")

# Provide the image path
image_path = 'thumb.jpg'
classify_image(image_path, model)
