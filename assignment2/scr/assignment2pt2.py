### Assignment 2 - Image classifier benchmark scripts
## Pt 2: Neural Network model
# By Line Stampe-Degn Møller
# Visual Analytics, Cultural Data Science

## TASKS:
# - Load either the MNIST_784 data or the CIFAR_10 data
# - Train a Neural Network model using the premade module in neuralnetwork.py
# - Print output to the terminal during training showing epochs and loss
# - Print the classification report to the terminal and save the classification report to out/nn_report.txt

## USAGE:

# To run this script:
# python3 scr/assignment2pt2.py

# IMPORTS:

# tf tools
import tensorflow as tf

# Path tools
import sys,os
sys.path.append(os.path.join(".."))

# File reading and writing
import csv

# Image processing
import cv2

# Neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10 # This is the data that we want to use 
from utils.neuralnetwork import NeuralNetwork

# Machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# LOAD THE CIFAR_10 DATA:
print("Currently loading cifar10 dataset - this may take a while")
(X_train, y_train), (X_test, y_test) = cifar10.load_data() # This takes a little time to run! 

# TRAIN A NEURAL NETWORK MODEL:
# Check shape:
shape = X_train.shape  # Four dimenssions - all the images, x-axis, y-axis, color
print("Shape of data:\n",shape)

labels = ["airplane",
         "automobile",
         "bird",
         "cat",
         "deer",
         "dog",
         "frog",
         "horse",
         "ship",
         "truck"]

# Convert to greyscale:
X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

# Normalize:
def minmax(data):
    X_norm = (data - data.min())/(data.max() - data.min())
    return X_norm

X_train_scaled = minmax(X_train_grey)
X_test_scaled = minmax(X_test_grey)

# Reshape:
nsamples, nx, ny = X_train_scaled.shape
X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

nsamples, nx, ny = X_test_scaled.shape
X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

print("[INFO] training network...")
input_shape = X_train_dataset.shape[1]
nn = NeuralNetwork([input_shape, 64, 10])
print(f"[INFO] {nn}")
nn.fit(X_train_dataset, y_train, epochs=10, displayUpdate=1)

predictions = nn.predict(X_test_dataset)

y_pred = predictions.argmax(axis=1)
report = classification_report(y_test.argmax(axis=1), y_pred, target_names=labels)

print("\nClassification report:\n",report)

# SAVE CLASSIFICATION REPORT
with open('out/cl_report_pt2.txt', 'w', encoding='UTF8') as f:
    f.write(report)

