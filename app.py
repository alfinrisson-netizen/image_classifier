import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("image_classifier.keras")

model = load_model()

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

st.title("Image Classifier")
st.write("Upload an image and let the AI predict its class.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Force RGB (3 channels)
    image = image.convert("RGB")

    st.image(image, caption="Uploaded Image")

    image = image.resize((32, 32))

    image_array = np.array(image, dtype=np.float32)

    image_array = image_array / 255.0

    # Debug: should print (32, 32, 3)
    st.write("Image shape:", image_array.shape)

    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)

    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")