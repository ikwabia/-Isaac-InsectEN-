
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('insect_edibility_classifier_v2.h5')

model = load_model()

st.title("🦋 Isaac InsectEN")

uploaded_file = st.file_uploader("Upload an insect image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((180, 180))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    label = "🟢 Edible" if prediction < 0.7 else "🔴 Non-Edible"

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence score: {prediction:.2f}")
