#  Wildfire Detection and Classification using Satellite Images with AI

##  Project Overview

This project aims to develop an AI-based solution for **early detection and classification of wildfires** using satellite images. Wildfires are a growing threat globally, causing irreversible environmental damage, economic loss, and human displacement. Detecting them early can significantly reduce their devastating impact.

By training a **Convolutional Neural Network (CNN)** model using labeled satellite images, this system can automatically classify whether a given image shows signs of wildfire or not. This project is part of the academic course *Discipline-Specific AI Project*.

---

##  Problem Statement

Wildfires spread rapidly and are often detected too late. Traditional detection methods (like human monitoring or manual interpretation of images) are time-consuming and inefficient.

> **Goal:** Build an AI model that can accurately classify satellite images into two categories:  
> - `Wildfire`  
> - `No Wildfire`

---

##  Objectives

- Collect and preprocess satellite images of wildfire-affected and unaffected regions.
- Train a deep learning model (ResNet50-based CNN) for binary classification.
- Evaluate and validate the model.
- Deploy the model for real-time wildfire detection and future scalability.

---

##  Technologies Used

- Python 
- TensorFlow / Keras
- Google Colab
- CNN with ResNet50
- Google Drive for data and model storage
- Matplotlib for performance visualization

---

## Dataset

- Satellite Images from NASA Firm and Kaggle Datasets
- Images are organized in folders:

---
##  How to Run
---
###  Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

---

## Step 2: Install dependencies and import libraries
---

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

---
## Step 3: Train and Save the Model
---

#Use the train.py script or Jupyter notebook provided to train the model and save it
model.save('/content/drive/MyDrive/wildfire_detection_model_resnet50.h5')

---

## Step 4: Load and Classify

---

model = load_model('/content/drive/MyDrive/wildfire_detection_model_resnet50.h5')

img_path = '/path/to/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_indices = {'no wildfire': 0, 'wildfire': 1}
labels = dict((v, k) for k, v in class_indices.items())
print(f"Prediction: {labels[np.argmax(prediction)]}")
