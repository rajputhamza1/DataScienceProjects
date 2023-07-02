#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import numpy as np
import cv2
import gdown
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint


# # Set the random seed for reproducibility

# In[2]:


random.seed(42)
np.random.seed(42)


# # Set the path to the dataset

# In[3]:


data_dir = 'C:\\Users\\rajpu\\Documents\\Python Scripts\\COVID CT\\Capstone_project-20230702T085249Z-001\\Capstone_project'


# # Set the desired image size

# In[4]:


image_size = (224, 224)


# 
# # Step 1: Load and preprocess the dataset

# In[5]:


def load_dataset():
    # Load COVID-19 positive images
    covid_dir = os.path.join(data_dir, 'COVID')
    covid_images = []
    for filename in os.listdir(covid_dir):
        img = cv2.imread(os.path.join(covid_dir, filename))
        img = cv2.resize(img, image_size)
        covid_images.append(img)

    # Load non-infected images
    non_infected_dir = os.path.join(data_dir, 'non-COVID')
    non_infected_images = []
    for filename in os.listdir(non_infected_dir):
        img = cv2.imread(os.path.join(non_infected_dir, filename))
        img = cv2.resize(img, image_size)
        non_infected_images.append(img)

    # Create labels (1 for COVID-19 positive, 0 for non-infected)
    labels = [1] * len(covid_images) + [0] * len(non_infected_images)

    # Combine images and labels
    images = covid_images + non_infected_images

    return np.array(images), np.array(labels)


# # Step 2: Data augmentation

# In[6]:


def apply_data_augmentation(X_train):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    return datagen


# # Step 3: Build and train the model

# In[7]:


from keras.preprocessing.image import ImageDataGenerator

def build_and_train_model(X_train, y_train, X_val, y_val):
    # Load pre-trained ResNet50 model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Set up data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    batch_size = 32

    # Create data generator from training set
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

    # Set up early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    model.fit(train_generator, steps_per_epoch=len(X_train) // batch_size, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping, checkpoint])

    return model


# # Step 5: Perform predictions and evaluate performance

# In[8]:


def evaluate_model(model, X_test, y_test):
    # Perform predictions
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).flatten()

    # Calculate performance metrics
    accuracy = np.mean(y_pred == y_test)
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
    f1_score = 2 * precision * recall / (precision + recall)

    # Print performance metrics
    print('Accuracy:', accuracy*100)
    print('Precision:', precision*100)
    print('Recall:', recall*100)
    print('F1-Score:', f1_score*100)


# ## Load and preprocess the dataset

# In[9]:


X, y = load_dataset()


# ## Split the dataset into train, validation, and test sets

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)


# ## Apply data augmentation

# In[11]:


datagen = apply_data_augmentation(X_train)


# ## Build and train the model

# In[12]:


model = build_and_train_model(X_train, y_train, X_val, y_val)


# ## Evaluate the model

# In[13]:


evaluate_model(model, X_test, y_test)

