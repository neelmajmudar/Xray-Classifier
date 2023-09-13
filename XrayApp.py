import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# Trained Model - 89% Accuracy
model = tf.keras.models.load_model('model.h5')

# Prediction Labels
class_labels = ['COVID-19', 'Normal', 'Pneumonia']

# Streamlit App
st.title("X-ray Image Classifier")

# Upload image
uploaded_image = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    #Predict using a button
    if st.button("Predict"):
        #Normalize and process the image
        img = load_img(uploaded_image, target_size=(256, 256), color_mode='grayscale')
        image_array = img_to_array(img)
        image_array = image_array / 255
        image_array = np.expand_dims(image_array, axis=0)
        
        #Predict
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        
        st.subheader("Prediction:")
        st.write(f"The X-ray Prediction Model Classifies this X-ray as: {class_labels[predicted_class]}")

        st.subheader("Model's Stats")
        #Display each plot
        plot_image_paths = ['ConfusionMatrixActualvPredicted.png', 'Training&ValidationAccuracy.png', 'Training&ValidationLoss.png']
        for image_path in plot_image_paths:
            st.image(image_path, caption=image_path, use_column_width=True)

