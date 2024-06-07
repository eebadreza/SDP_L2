import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    # print(type(image))

    # convert image to numpy array
    image_array = np.asarray(image)
    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # print(type(image), type(image_array), type(normalized_image_array))
    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    # prediction = model.predict(data)
    # index = np.argmax(prediction)
    # index = 0 if prediction[0][0] > 0.95 else 1
    # class_name = class_names[index]
    # confidence_score = prediction[0][index]

    # prediction = model.predict(data)
    # index = np.argmax(prediction)
    # class_name = class_names[index]
    # confidence_score = prediction[0][index]

    # Predict the probabilities for each class
    prediction = model.predict(data)
    # Get the index of the class with the highest probability
    index = np.argmax(prediction)

    # Get the class name using the index
    if index == 0:
        class_name = class_names[index]
    else:
        class_name = 'DR, ' + class_names[index]

    # Get the confidence score of the predicted class
    confidence_score = prediction[0][index]
    # Print the results
    print(f"Predicted class: {class_name}")
    print(f"Confidence score: {confidence_score}")
    print(prediction, index, class_name, confidence_score)
    return class_name, confidence_score
