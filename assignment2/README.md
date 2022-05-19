# Assignment 2 - Image classifier benchmark scripts
**Visual Analytics, Cultural Data Science**

By: Line Stampe-Degn Møller

Contributors: None

Link to this repository: (https://github.com/linesdmoller/cds_vis_line/tree/main/assignment2)

## PROJECT DESCRIPTION:
*This project is assignment 2 in the supplementary course, 'Visual Analytics' in 'Cultural Data Science', at Aarhus University.*
- Link to assignment description: (https://github.com/CDS-AU-DK/cds-visual/blob/main/assignments/assignment2.md)

The assignment consists of two projects:
1. Pt 1: Logistic regression model
2. Pt 2: Neural Network model

The two projects and their output is saved in the same folder system as 'assignment2pt1.py' and 'assignment2pt2.py' in the 'scr' folder, and 'cl_report_pt1.txt' and 'cl_report_pt2.txt' in the 'out' folder. 

**Pt 1 task outline:**
- Load either the MNIST_784 data or the CIFAR_10 data (I choose the CIFAR_10 dataset)
- Train a Logistic Regression model using scikit-learn
- Print the classification report to the terminal and save the classification report to out/lr_report.txt

**Pt 2 task outline:**
- Load either the MNIST_784 data or the CIFAR_10 data (I choose the CIFAR_10 dataset)
- Train a Neural Network model using the premade module in neuralnetwork.py
- Print output to the terminal during training showing epochs and loss
- Print the classification report to the terminal and save the classification report to out/nn_report.txt

## METHODS:
The first part of the Python script in the two projects is the same. This is where i load and preprocess the data. Here, I first load the CIFAR_10 dataset. I then add labels, convert to greyscale, normalize and reshape the date.

**Assignment 2 pt 1: Logistic regression model**

This project uses scikit-learn's Logistic Regression model to make classification predictions on image data.

**Assignment 2 pt 2: Neural Network model**

This project uses a multilayered feedforward neural network, written in nympy, designed by Ross Deans Kristensen-McLachlan, to make classification predictions on image data. The network structure can be found in the 'utils' folder, under 'neuralnetwork.py'. 

## USAGE:
To run each of the two scripts in the terminal, navigate to the folder outside the 'src' folder and run, one of the two:
- python3 scr/assignment2pt1.py
- python3 scr/assignment2pt2.py

## DISCUSSION OF RESULTS:
**Assignment 2 pt 1: Logistic regression model**

The output of this project is; 
- A txt of the classification report, 'cl_report_pt1.txt' (See [Link](https://github.com/linesdmoller/cds_vis_line/blob/main/assignment2/out/cl_report_pt1.txt)).

The classification report shows an f1 score of around 0.31. This means, that the model predicts classes with 31% accuracy. This is not a great model but the approach is a simple example of an image classifyer using scikit-learn's Logistic Regression model. According to the classification report, the top 4 classes that the model is best at predicting are;
1. 'truck'
2. 'ship'
3. 'automobile'
4. 'airplane'

**Assignment 2 pt 2: Neural Network model**

The output of this project is; 
- A txt of the classification report, 'cl_report_pt2.txt' (See [Link](https://github.com/linesdmoller/cds_vis_line/blob/main/assignment2/out/cl_report_pt2.txt)).

The classification report shows an f1 score of around 0.37. This means, that the model predicts classes with 37% accuracy. This is not a great model but the approach is a simple example of an image classifyer using a multilayered feedforward neural network. According to the classification report, the top 4 classes that the model is best at predicting are;
1. 'ship'
2. 'truck'
3. 'automobile'
4. 'horse'
