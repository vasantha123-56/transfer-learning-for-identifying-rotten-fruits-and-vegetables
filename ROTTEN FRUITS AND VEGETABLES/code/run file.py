import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import Image
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# THEMING
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
def apply_custom_theme(theme):
    if theme == "Dark":
        css = """
        <style>
        .stApp {background-color: #0e1117; color: white;}
        h1, h2, h3, h4, h5, h6, p, span, label {color: white !important;}
        </style>
        """
    else:
        css = """
        <style>
        .stApp {background-color: white; color: black;}
        h1, h2, h3, h4, h5, h6, p, span, label {color: black !important;}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)
apply_custom_theme(theme)

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Dataset Info","Prediction"])
DATASET_URL = "https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

if app_mode == "Home":
    st.markdown(
        "<span style='color:#000000; font-size:2.5em; font-weight:bold;'>IDENTIFYING ROTTEN FRUITS & VEGETABLES RECOGNITION SYSTEM</span>",
        unsafe_allow_html=True
    )

    image_path = "background img/Background_img.jpg"  # Make sure the image is in the same directory as this script
    img_base64 = get_base64_of_bin_file(image_path)
    home_bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .stApp * {{
        color: green !important;
    }}
    </style>
    """
    st.markdown(home_bg_css, unsafe_allow_html=True)

#About Project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango,coconut")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")
  
#data set info  
elif app_mode == "Dataset Info":
    st.header("Dataset Information & Download")
    st.markdown(f"""
    **Dataset Used:**  
    [Fruit and Vegetable Disease - Healthy vs Rotten (Kaggle)]({DATASET_URL})

    - **Kaggle Download Instructions:**  
      1. Go to [the dataset page]({DATASET_URL})  
      2. Download and unzip the dataset manually.
      3. You can then upload images or zipped folders for further processing in this app.
    """)
    st.info("To use this dataset for training or prediction, download it from Kaggle as a ZIP and upload via the Upload Dataset page below.")


elif app_mode == "Prediction":
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('trained_model/trained_model.h5')
    model = load_model()
    with open('labels/labels.txt') as f:
        labels = [line.strip() for line in f]

    st.title("Rotten Fruit & Vegetables Classifier")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Show Image button
        if st.button("Show Image"):
            st.image(img, caption="Uploaded Image", use_column_width=True, width=400)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_label = labels[predicted_index]
            confidence = predictions[0][predicted_index] * 100
            st.success(f"Prediction: {predicted_label} ({confidence:.2f}%)")
            
            
            
            
            
            
            # OUTPUT VIDEO LINK :  https://drive.google.com/file/d/1G80UjwIbJrsQKc0SE1Vf2rf2BOWY3aHa/view?usp=drivesdk
            
            
            
        ###     Final Report  ==  https://drive.google.com/file/d/19kmX94Y3Nn-fsX24Y8OpDFUrT1GlG7Ci/view?usp=drivesdk
    
    