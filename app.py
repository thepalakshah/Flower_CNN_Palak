import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Flower CNN Classifier")
st.write("Upload an image to classify the flower.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Load your trained model
    model = load_model("your_model.h5")  # replace with your model path

    # Preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(64, 64))  # adjust size
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    st.write("Prediction:", np.argmax(prediction))
