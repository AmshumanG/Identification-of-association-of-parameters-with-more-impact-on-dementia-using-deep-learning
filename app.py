import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Only keep this import
import numpy as np
from tensorflow.keras.models import load_model  # Correct import for TensorFlow 2.x

# Load the model
model = load_model('./Model/alzheimer .h5', compile=False)

# Dictionary to map predictions
lab = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}

def processed_img(img_path):
    # Load and preprocess image
    img = load_img(img_path, target_size=(224, 224))  # Remove 3rd dimension from target_size
    img = img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model prediction
    
    # Model prediction
    answer = model.predict(img)
    y_class = np.argmax(answer, axis=-1)  # Get the class with the highest probability
    
    # Mapping the prediction to the label
    res = lab[y_class[0]]  # Use y_class[0] because it's an array with a single value
    return res

def run():
    # Streamlit UI
    img1 = Image.open('./meta/logo1.png')
    img1 = img1.resize((350, 350))
    st.image(img1, use_column_width=False)
    st.title("Impact of demantia parameters using Deep Learning")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based on "KAGGLE BASED DATASET"</h4>''',
                unsafe_allow_html=True)

    # Image file uploader
    img_file = st.file_uploader("Choose an Image of MRI", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file, use_column_width=False)
        
        # Save the uploaded image to a directory
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Predict on the image when the button is pressed
        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success(f"Predicted: {result}")

# Run the app
run()
