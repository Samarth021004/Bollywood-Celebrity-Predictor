#!pip install streamlit

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

# Initialize MTCNN detector and ResNet50 model
detector = MTCNN()
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed feature list and filenames
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

# Function to save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

# Function to extract features from image
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if len(results) > 0:  # Ensure that at least one face is detected
        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]

        # Extract its features
        image = Image.fromarray(face)
        image = image.resize((224, 224))

        face_array = np.asarray(image)
        face_array = face_array.astype('float32')

        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()
        return result
    else:
        return None

# Function to recommend the most similar celebrity
def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# Streamlit app title
st.title('Which Bollywood Celebrity Are You?')

# Upload image
uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # Save the uploaded image
    if save_uploaded_image(uploaded_image):
        # Load the image for display
        display_image = Image.open(uploaded_image)

        # Extract features
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        
        if features is not None:
            # Recommend the most similar celebrity
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.header('Your uploaded image')
                st.image(display_image)

            with col2:
                st.header("Seems like " + predicted_actor)
                st.image(filenames[index_pos], width=300)
        else:
            st.error("No face detected in the image. Please try another image.")