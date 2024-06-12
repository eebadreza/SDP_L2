# import cv2
import os
import streamlit as st
import numpy as np
from PIL import Image
from util import classify, set_background
import tensorflow as tf

# set_background('blk.jpeg')
# print(os.getcwd())
# set title
st.title('Diabetic Retinopathy classification')
# set header
st.header('Please upload a Retinal Scan Image')
# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
# model = tf.keras.models.load_model('/Users/apple/Downloads/SDP/pretrained_models/B/model_weighted_avg.keras', custom_objects=dict(CustomLayer=CustomLayer))
# model = tf.keras.models.load_model('/Users/apple/Downloads/SDP/pretrained_models/B/model_weighted_avg.keras', custom_objects={ "TransformerEncoderLayer": TransformerEncoderLayer, "PositiionalEmbedding": PositiionalEmbedding, "TransformerDecoder": TransformerDecoder})
################################
model_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained_models', 'B', 'model_mobilenet.keras')

# Normalize the path to avoid issues with different operating systems
model_path = os.path.normpath(model_path)
print("Full path to the model file:", model_path)

# Check if the model file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the model
model = tf.keras.models.load_model(model_path)
################################
# model = tf.keras.models.load_model('../pretrained_models/B/model_mobilenet.keras')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
print(class_names)

# display image
if file is not None:
    # image = Image.open(file).convert('RGB')
    image = Image.open(file)
    st.image(image, use_column_width=True)
    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
