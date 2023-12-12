import base64

# util.py

import numpy as np
from PIL import ImageOps, Image
import base64
import streamlit as st

def classify(image, model, class_names):
    # Resize image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Convert image to grayscale
    gray_image_array = ImageOps.grayscale(image)

    # Convert grayscale image to numpy array
    gray_image_array = np.asarray(gray_image_array)

    # Normalize grayscale image
    normalized_gray_image_array = (gray_image_array / 127.5) - 1

    # Set model input
    data = np.ndarray(shape=(1, 224, 224, 1), dtype=np.float32)
    data[0, :, :, 0] = normalized_gray_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
    
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
   
