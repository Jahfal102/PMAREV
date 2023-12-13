import streamlit as st
from keras.optimizers import Adam
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


from util import classify, set_background

# Set background
set_background('./BG/bg.jpg')

# Set title
st.title('Pneumonia covid classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('./model/PMA4.h5')

# Compile the model
optimizer = Adam(lr=0.01)  # Adjust the learning rate as needed
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Load class names
def diagnosis(file, model, IMM_SIZE):
    # Load and preprocess the image
    img = image.load_img(file, target_size=(IMM_SIZE, IMM_SIZE), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]

    # Predict the diagnosis
    predicted_probabilities = model.predict(img_array)
    predicted_class = np.argmax(predicted_probabilities, axis=-1)[0]

    # Map the predicted class to the diagnosis
    diagnosis_mapping = {0: "Normal", 1: "Covid", 2: "Viral Pneumonia"}
    predicted_diagnosis = diagnosis_mapping[predicted_class]

    return predicted_diagnosis

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
