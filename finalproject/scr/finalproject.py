### Final Project: Rock, Paper, Scissors classifier
# By Line Stampe-Degn Møller
# Visual Analytics, Cultural Data Science

## USAGE:

# To run this script:
# python3 scr/finalproject.py

##################################################################

# IMPORTS
# os
import os

# tf tools
import tensorflow as tf

# Randomizer
import random

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization, 
                                     GlobalAveragePooling2D, 
                                     Conv2D, 
                                     MaxPooling2D)

# models
from tensorflow.keras.models import Sequential

# generic model object
from tensorflow.keras.models import Model

# Plot model
from tensorflow.keras.utils import plot_model

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
    plt.savefig("out/his_plt.png")

# FIND FILES
for dirname, _, filenames in os.walk("in"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break
    break

directory = "in"
print(os.listdir(directory))

# Define labels
labels = ['paper','scissors','rock']
labelcount = len(labels)

# Save all images to an array, normalize and shuffle them
def input_target_split(train_dir,labels):
    dataset = []
    count = 0
    for label in labels:
        folder = os.path.join(train_dir,label)
        for image in os.listdir(folder):
            img=load_img(os.path.join(folder,image), target_size=(150,150))
            img=img_to_array(img)
            img=img/255  #normalize
            dataset.append((img,count))
        print(f'\rCompleted: {label}',end='')
        count+=1
    random.shuffle(dataset)
    X, y = zip(*dataset)
    
    return np.array(X),np.array(y)

X, y = input_target_split(directory,labels)

# Number of folders and their content
np.unique(y,return_counts=True)

# SPLIT IMAGES into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

# DEFINE MODEL
# Create model
model = Sequential()

# First convolutional layer and maxpooling
model.add(Conv2D(32, 
                 (3, 3), 
                 input_shape=(150, 150, 3), 
                 activation='relu'))
model.add(MaxPooling2D(2, 2))

# Second convolutional layer and maxpooling
model.add(Conv2D(32,
                 (3, 3), 
                 activation = 'relu'))
model.add(MaxPooling2D(2, 2))

# Fully-connected classification layer
model.add(Flatten())
model.add(Dense(units=512, 
                activation='relu'))
model.add(Dense(units=3, 
                activation='softmax'))

# Compile model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
sgd = SGD(learning_rate=lr_schedule)

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.summary()

modelplot = plot_model(model, show_shapes=True, show_layer_names=True, to_file='out/visualised_model.png')

# Data augmentation (slightly distorting images for more diverse images)
datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             zoom_range=0.2,
                             rotation_range=20,
                             shear_range=0.1,
                             fill_mode="nearest")

testgen = ImageDataGenerator()

# Fit to data set
datagen.fit(X_train)
testgen.fit(X_test)

y_train = np.eye(labelcount)[y_train]
y_test = np.eye(labelcount)[y_test]

# TRAIN MODEL
H = model.fit(datagen.flow(X_train, 
                           y_train, 
                           batch_size=32), 
              validation_data=testgen.flow(X_test, 
                                           y_test, 
                                           batch_size=32), 
              epochs=50)

# PLOT (and save) LOSS AND ACCURACY
plot_history(H, 50)

# MAKE CLASSIFICATION REPORT
predictions = model.predict(X_test, batch_size=128)
report = classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labels)

# SAVE CLASSIFICATION REPORT
with open('out/cl_report.txt', 'w', encoding='UTF8') as f:
    f.write(report)
