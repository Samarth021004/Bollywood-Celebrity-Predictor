# !pip install mtcnn==0.1.0
# !pip install tensorflow==2.3.1
# !pip install keras==2.4.3
# !pip install keras-vggface==0.6
# !pip install keras_applications==1.0.8
'''import os
import pickle

actors = os.listdir('data')
print(actors) // gives a list of all subfolders in main folder 

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join('data',actor)):
        filenames.append(os.path.join('data',actor,file))
print(filenames) // all files will be outputted
print(len(filenames)

pickle.dump(filenames,open('filenames.pkl','wb'))'''

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import pickle
from tqdm import tqdm

# Load the filenames from a pickle file
with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

# Initialize the ResNet50 model (similar to VGGFace)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def feature_extractor(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    # Extract features using the model
    result = model.predict(preprocessed_img).flatten()
    return result

# Extract features for each file in the list
features = []
for file in tqdm(filenames):
    features.append(feature_extractor(file, model))
    #print(result.shape) shows 2048 features have been extracted 

# Save the extracted features to a pickle file
with open('embedding.pkl', 'wb') as f:
    pickle.dump(features, f)
    