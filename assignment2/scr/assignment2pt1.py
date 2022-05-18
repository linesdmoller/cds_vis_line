### Assignment 2 - Image classifier benchmark scripts
## Pt 1: Logistic regression model
# By Line Stampe-Degn Møller
# Visual Analytics, Cultural Data Science

## TASKS:
# - Load either the MNIST_784 data or the CIFAR_10 data
# - Train a Logistic Regression model using scikit-learn
# - Print the classification report to the terminal and save the classification report to out/lr_report.txt

## USAGE:

# To run this script:
# python3 scr/assignment2pt1.py

# IMPORTS:

# Path tools
import sys,os
sys.path.append(os.path.join(".."))

# tf tools
import tensorflow as tf

# File reading and writing
import csv

# Image processing
import cv2

# Neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10 # This is the data that we want to use 

# Machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# LOAD THE CIFAR_10 DATA:

print("Currently loading cifar10 dataset - this may take a while")
(X_train, y_train), (X_test, y_test) = cifar10.load_data() # This takes a little time to run! 


# TRAIN A LOGISTIC REGRESSION MODEL USING SCIKIT-LEARN:
# Check shape:
shape = X_train.shape  # Four dimenssions - all the images, x-axis, y-axis, color
print("Shape of data:\n",shape)

# Labels
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

# Convert to np array
X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])

X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

# Normalize
def minmax(data):
    X_norm = (data - data.min())/(data.max() - data.min())   
    return X_norm
X_train_scaled = minmax(X_train)
X_test_scaled = minmax(X_test)

# Reshape data
nsamples, nx, ny = X_train_scaled.shape
X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

nsamples, nx, ny = X_test_scaled.shape
X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

# Logistic regression classifyer
clf = LogisticRegression(penalty = 'none',
                         tol = 0.1,
                         solver = 'saga',
                         multi_class = "multinomial").fit(X_train_dataset, y_train)

# Get predictions and make a classification report
y_pred = clf.predict(X_test_dataset)
report = classification_report(y_test, y_pred, target_names=labels)

# Print report in console
print("\nClassification report:\n",report)

# SAVE CLASSIFICATION REPORT
with open('out/cl_report_pt1.txt', 'w', encoding='UTF8') as f:
    f.write(report)
