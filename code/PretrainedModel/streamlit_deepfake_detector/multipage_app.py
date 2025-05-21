# -------------------
# IMPORTS
# -------------------
import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image, ImageOps
from streamlit_image_select import image_select
from tensorflow.keras.models import model_from_json
import time

# Set page configuration FIRST
st.set_page_config(layout="wide")


@st.cache_resource()
def load_model():
    base_path = os.path.dirname(__file__)  # directory where multipage_app.py is
    json_path = os.path.join(base_path, "dffnetv2B0.json")
    h5_path = os.path.join(base_path, "dffnetv2B0.h5")

    with open(json_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(h5_path)
    return model


# function to preprocess an image and get a prediction from the model
def get_prediction(model, image):
    open_image = Image.open(image)
    resized_image = open_image.resize((256, 256))
    np_image = np.array(resized_image)
    reshaped = np.expand_dims(np_image, axis=0)

    predicted_prob = model.predict(reshaped)[0][0]

    if predicted_prob >= 0.5:
        return "Real", predicted_prob
    else:
        return "Fake", 1 - predicted_prob

# generate selection of sample images 
@st.cache_data()
def load_images():
    real_images = ["images/Real/" + x for x in os.listdir("images/Real/")]
    fake_images = ["images/Fake/" + x for x in os.listdir("images/Fake/")]
    image_library = real_images + fake_images
    image_selection = np.random.choice(image_library, 20, replace=False)
    return image_selection

# -------------------
# APP MODES
# -------------------

def game_mode(classifier, images):
    st.header("Game Mode")
    st.subheader("Can you beat the model?")

    selected_image = image_select(
        "Click on an image below to guess if it is real or fake:", 
        images,
        return_value="index")

    prediction = get_prediction(classifier, images[selected_image])
    true_label = 'Fake' if 'fake' in images[selected_image].lower() else 'Real'

    st.subheader("Is this image real or fake?")
    st.image(images[selected_image])

    if st.button("It's Real"):
        st.text("You guessed:")
        st.subheader("*Real*")
        st.text("The Deepfake Detector model guessed...")
        time.sleep(1)
        st.subheader(f"*{prediction}*")
        st.text("The truth is...")
        time.sleep(1)
        st.subheader(f"***It's {true_label}!***")

    if st.button("It's Fake"):
        st.text("You guessed:")
        st.subheader("*Fake*")
        st.text("The Deepfake Detector model guessed...")
        time.sleep(1)
        st.subheader(f"*{prediction}*")
        st.text("The truth is...")
        time.sleep(1)
        st.subheader(f"***It's {true_label}!***")

def detector_mode(classifier):
    st.header("Detector Mode")
    st.subheader("Upload an Image to Make a Prediction")

    uploaded_image = st.file_uploader("Upload your own image to test the model:", type=['jpg', 'jpeg'])

    if uploaded_image is not None:
        st.image(uploaded_image)

        label, confidence = get_prediction(classifier, uploaded_image)
        color = "green" if label == "Real" else "crimson"
        st.markdown(f"<h2 style='color:{color};'>{label}, Confidence: {confidence:.2f}</h2>", unsafe_allow_html=True)


# -------------------
# MAIN FUNCTION
# -------------------
def main():
    st.title("Deepfake Detector:")

    # load model and sample images
    classifier = load_model()
    images = load_images()

    # sidebar for mode selection
    page = st.sidebar.selectbox('Select Mode', ['Detector Mode', 'Game Mode']) 

    # page routing
    if page == 'Game Mode':
        game_mode(classifier, images)
    else:
        detector_mode(classifier)

# -------------------
# RUN APP
# -------------------
if __name__ == "__main__":
    main()
