### Assignment 3 - Transfer learning + CNN classification
# By Line Stampe-Degn Møller
# Visual Analytics, Cultural Data Science

## TASKS:
# - Load the CIFAR10 dataset
# - Use VGG16 to perform feature extraction
# - Train a classifier
# - Save plots of the loss and accuracy
# - Save the classification report

## USAGE:

# To run this script:
# python3 scr/assignment3.py


# IMPORTS
# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt

# csv
import csv 

# PLOTTING FUNCTION:
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    # SAVE PLOTS OF LOSS AND ACCURACY
    plt.savefig('out/his_plt.png')
    
# LOAD THE CIFAR_10 DATA:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the Cifar_10 dataset
X_train= X_train.astype("float") / 255.
X_test = X_test.astype("float") / 255.

# integers to one-hot vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# initialize label names for CIFAR-10 dataset
labelNames = ["airplane", "automobile", 
              "bird", "cat", 
              "deer", "dog", 
              "frog", "horse", 
              "ship", "truck"]

# load model without classifier layers
model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(32, 32, 3))

# mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False

# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu')(flat1)
output = Dense(10, activation='softmax')(class1)

# define new model
model = Model(inputs=model.inputs, 
              outputs=output)

# COMPILE:
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
sgd = SGD(learning_rate=lr_schedule)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TRAIN:
H = model.fit(X_train, y_train, 
              validation_data=(X_test, y_test), 
              batch_size=128,
              epochs=10,
              verbose=1)

#PLOT (and save) LOSS AND ACCURACY
plot_history(H, 10)

# MAKE CLASSIFICATION REPORT
predictions = model.predict(X_test, batch_size=128)
report = classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames)

# SAVE CLASSIFICATION REPORT
with open('out/cl_report.txt', 'w', encoding='UTF8') as f:
    f.write(report)
