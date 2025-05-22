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

st.markdown("""
    <style>
    /* Center all content and control uploader width */
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }

    /* Shrink the uploader box width */
    .stFileUploader {
        max-width: 850px;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

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
    base_path = os.path.dirname(__file__)  # Get the directory of the current script

    real_dir = os.path.join(base_path, "images", "Real")
    fake_dir = os.path.join(base_path, "images", "Fake")

    # Ensure only files are listed (skip subdirs or hidden files)
    real_images = [os.path.join("images/Real", x) for x in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, x))]
    fake_images = [os.path.join("images/Fake", x) for x in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, x))]

    image_library = real_images + fake_images

    # Safety check: ensure at least 20 images available
    sample_size = min(20, len(image_library))
    image_selection = np.random.choice(image_library, sample_size, replace=False)

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
    st.markdown("<h3 style='text-align: center;'>Upload an Image to Make a Prediction</h3>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader(" ", type=['jpg', 'jpeg'])

    if uploaded_image is not None:
        st.image(uploaded_image)

        label, confidence = get_prediction(classifier, uploaded_image)
        color = "green" if label == "Real" else "crimson"
        st.markdown(f"<h2 style='color:{color};'>{label}, Confidence: {confidence:.2f}</h2>", unsafe_allow_html=True)


# -------------------
# MAIN FUNCTION
# -------------------
def main():
    st.markdown("""
<h1 style='text-align: center;'>
    🕵️‍♂️ Deepfake Image Detector:
</h1>
""", unsafe_allow_html=True)
  
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
