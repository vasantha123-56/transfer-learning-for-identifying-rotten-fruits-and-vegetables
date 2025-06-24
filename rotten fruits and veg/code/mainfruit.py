import streamlit as st
import tensorflow as tf
import numpy as np
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

##Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        
        # Get model predictions (probabilities)
        def model_prediction_with_probs(test_image):
            model = tf.keras.models.load_model("trained_model/trained_model (1).h5")
            image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr]) #convert single image to batch
            predictions = model.predict(input_arr)  # shape: (1, num_classes)
            return predictions[0] # return probabilities for each class

        # Reading Labels
        with open("labels/labels/labels.txt") as f:
            content = f.readlines()
        labels = [i.strip() for i in content]

        probs = model_prediction_with_probs(test_image)
        result_index = np.argmax(probs)
        predicted_label = labels[result_index]
        predicted_prob = probs[result_index] * 100

        st.success("Model is Predicting it's a {}".format(predicted_label))
        st.info(" Fresh & Healthy : {:.2f}%".format(predicted_prob))

        # If label format is like 'Apple_Good', 'Banana_Bad', show good/bad breakdown
        if "_" in predicted_label:
            fruit_name = predicted_label.split("_")[0]
            good_indexes = [i for i, l in enumerate(labels) if l == f"{fruit_name}_Good"]
            bad_indexes = [i for i, l in enumerate(labels) if l == f"{fruit_name}_Bad"]
            good_prob = sum(probs[i] for i in good_indexes) * 100 if good_indexes else None
            bad_prob = sum(probs[i] for i in bad_indexes) * 100 if bad_indexes else None

            if good_prob is not None:
                st.write("Good: {:.2f}%".format(good_prob))
            if bad_prob is not None:
                st.write("Bad: {:.2f}%".format(bad_prob))


    
    
    
    
    #####   Download the trained_model.h5 file == https://drive.google.com/file/d/17pG_EQ3o1-YreRYV09VXzjA6w7b3-fpZ/view?usp=drivesdk
    
    ####    Labels text file  ==  https://drive.google.com/file/d/19s8mp9K3j-XZQXJ1HHWoN2hzZbAI6o60/view?usp=drivesdk
    
    ####    Video Link ==  https://drive.google.com/file/d/1a5jZIL8ygLnEuCPbm7IiXvpAvcAQjFSq/view?usp=drivesdk
    
    ###     Final Report  ==  https://drive.google.com/file/d/19kmX94Y3Nn-fsX24Y8OpDFUrT1GlG7Ci/view?usp=drivesdk
    
    