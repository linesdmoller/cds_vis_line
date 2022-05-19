# Assignment 3 - Transfer learning + CNN classification
**Visual Analytics, Cultural Data Science**

By: Line Stampe-Degn Møller

Contributors: None

Link to this repository: (https://github.com/linesdmoller/cds_vis_line/tree/main/assignment3)

## PROJECT DESCRIPTION:
*This project is assignment 3 in the supplementary course, 'Visual Analytics' in 'Cultural Data Science', at Aarhus University.*
- Link to assignment description: (https://github.com/CDS-AU-DK/cds-visual/blob/main/assignments/assignment3.md)

**Task outline:**
- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier
- Save plots of the loss and accuracy
- Save the classification report

## METHODS:
In this project, I use Load the CIFAR10 dataset. I then use Tensorflow's 'VGG16' model to perform feature extraction and add an additional classifier layers using Tensorflow's 'Keras' layers. I use scikit-learn's 'LabelBinarizer' to binarize integers to one-hot vectors in the 'y_train' and 'y_test'. I the compile and train the classifier model. Finally, I create and save a png of the history plots of the loss and accuracy curves and a txt file of the classification report.

## USAGE:

To run this script in the terminal, navigate to the folder outside the 'scr' folder and run:

python3 scr/assignment3.py

## DISCUSSION OF RESULTS:
The output of this project is;

A txt file containing the classification report (see [Link](https://github.com/linesdmoller/cds_vis_line/blob/main/assignment3/out/cl_report.txt)).
A png file containg the history plot from the training session (see [Link](https://github.com/linesdmoller/cds_vis_line/blob/main/assignment3/out/his_plt.png))



