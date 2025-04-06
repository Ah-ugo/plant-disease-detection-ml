import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load pre-trained model
model = load_model('plant_disease_model.keras')

# Define class labels
labels = ['Healthy', 'Powdery', 'Rust']

st.title("Plant Disease Classification")
st.write("Upload a plant leaf image to classify whether it is Healthy, has Powdery mildew, or Rust.")

# Display training accuracy plots
st.image('training_history.png', caption="Training History", use_column_width=True)

def preprocess_image(image):
    # Convert PIL Image to numpy array if it's not already
    image = np.array(image)

    # Resize the image to match training input size
    image = cv2.resize(image, (224, 224))

    # Convert from RGB to BGR - IMPORTANT CORRECTION
    # In your training code you used cv2.imread (which loads as BGR) then converted to RGB
    # But in Streamlit, PIL loads as RGB, so we need the opposite conversion
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert back to RGB to match the training process
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Expand dimensions and normalize
    image = np.expand_dims(image, axis=0) / 255.0
    return image


def predict_disease(image):
    processed_image = preprocess_image(image)

    # Add debugging output
    st.write(f"Image shape: {processed_image.shape}")
    st.write(f"Image min/max values: {processed_image.min()}/{processed_image.max()}")

    prediction = model.predict(processed_image)

    # Show prediction probabilities
    st.write("Prediction probabilities:")
    for i, label in enumerate(labels):
        st.write(f"{label}: {prediction[0][i] * 100:.2f}%")

    predicted_class = labels[np.argmax(prediction)]
    return predicted_class


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result = predict_disease(image)
        st.success(f'Predicted class: {result}')

        # Display confidence
        confidence = model.predict(preprocess_image(image))
        st.write("Confidence scores:")
        for i, label in enumerate(labels):
            st.progress(float(confidence[0][i]))
            st.write(f"{label}: {float(confidence[0][i]) * 100:.2f}%")