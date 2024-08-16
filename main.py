import matplotlib.pyplot as plt 
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import pickle

#set up
st.set_page_config(page_title="Action recognition")

class_names = pd.read_csv('./kinetics_labels.csv') 

Resnet50 = tf.keras.models.load_model('./resnet50.h5')
EfficientNetV2S = tf.keras.models.load_model('./efficientnetv2s.h5')

#example
st.header("Example")
example_images = ['./images/example1.jpg', './images/example2.jfif', './images/example3.jpg']

cols = st.columns(3)
for i, img_path in enumerate(example_images):
    with cols[i]:
        image = plt.imread(img_path)
        image_resized = cv2.resize(image, (224, 224))
        image_batch = np.expand_dims(image_resized, axis=0)
        
        preds1 = Resnet50.predict(image_batch)
        preds2 = EfficientNetV2S.predict(image_batch)
        preds = (preds1 + preds2) / 2.0
       
        top_5_indices_keras = np.argsort(preds[0])[-5:][::-1] #sort và return index, lấy 5 hạng đầu, reverse

        keras = [class_names.loc[int(i), 'name'] for i in top_5_indices_keras]
        st.image(img_path, caption=f"Top-5 accuracy: {', '.join(keras)}")

#upload images and predict
uploaded_files = st.file_uploader(
    "Upload some images to predict", accept_multiple_files=True
)

def helper(uploaded_file):        
    image = plt.imread(uploaded_file)  
    image_resized = cv2.resize(image, (224, 224))
    image_batch = np.expand_dims(image_resized, axis=0)
        
    preds1 = Resnet50.predict(image_batch)
    preds2 = EfficientNetV2S.predict(image_batch)
    preds_keras = (preds1 + preds2) / 2.0
    
    top_5_indices_keras = np.argsort(preds_keras[0])[-5:][::-1]
    keras = [class_names.loc[int(i), 'name'] for i in top_5_indices_keras]
    
    st.image(image_resized, caption=f"Top-5 accuracy: {', '.join(keras)}")
    
if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        if i % 3 == 0:
            cols = st.columns(3)
        with cols[i % 3]:
            helper(uploaded_file)
