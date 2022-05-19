# Final Project - Rock, Paper, Scissors
**Visual Analytics, Cultural Data Science**

By: Line Stampe-Degn Møller

Contributors: None

Link to this repository: (https://github.com/linesdmoller/cds_vis_line/tree/main/finalproject)

## PROJECT DESCRIPTION:
*This project is a self-assigned project submitted as a final project in the supplementary course, 'Visual Analytics' in 'Cultural Data Science', at Aarhus University.*

**Goal:**
The goal of this project is to build and train a Convolutional Neural Network model to predict classes of the hand gestures; 'rock', 'paper' and 'sciccors', based on a dataset of multiple images illustrating the same three hand gestures (according to the commonly known game; 'Rock, Paper, Scisssors'). I then want to evaluate the model's performance by making and saving a png of the history plot ('loss' and 'accuracy' curve) from the training session and a txt of the classification report.

## METHODS:
The dataset used in this project stems from; (https://www.kaggle.com/code/quadeer15sh/tf-keras-cnn-99-accuracy/data).

I approach the goal of this project by using TensorFlow's keras model to build the architecture of a Convolutional Neural Network. The model consists of 2 convolutional layers and one fully-connected classification layer. In order to train a better model, I use data augmentation techniques to slightly distort the images in the dataset to make them more diverse images (i.e. flipping, shifting, zooming, rotating, etc. the images). I train the model using 50 epochs. However, for a quick run of the script, 5 or 10 epochs also results in an acceptable accuracy in the model's predictions. 

## USAGE:
In order to reporduce this project, one must first download the input dataset from here; [Link](https://www.kaggle.com/code/quadeer15sh/tf-keras-cnn-99-accuracy/data), and add the data to the 'in' folder in this project. One might also need to delete the blank file.

To run this script in the terminal, navigate to the folder outside the 'scr' folder and run:

python3 scr/finalproject.py

## DISCUSSION OF RESULTS:
The output of this project is;
- A txt file containing the classification report (see [Link](https://github.com/linesdmoller/cds_vis_line/blob/main/finalproject/out/cl_report.txt)).
- A png file containg the history plot from the training session (see [Link](https://github.com/linesdmoller/cds_vis_line/blob/main/finalproject/out/his_plt.png))

The trained model performs really well - it has an f1 score of 0.99. This means that the model predictions are about 99% accurate. Test runs with fewer and more epochs suggest that the model performs better the more epochs it is train with. All test runs as well as the final results show, that the class, 'rock', is the class that the model is best at predicting. The f1 score of the 'rock' class in the final results is actually 1, meaning 100% (this is of course rounded up). The history plots show a rapit decrease in the loss curve and a rapid increase in the accuracy curve over the first 10 epochs. After that, the curves flatten out. This indicates that the model learns most of it's knowledge about how to solve the classification task over the first 10 epochs.

For further developement of this project, one could apply the model to a totally new image or live web cam to try to interpret the hand gesture in a game of 'Rock, Paper, Scissors'. The program could then be coded to counter play a random hand gesture and calculate the winner, so the program will be a game of man against machine in 'Rock, Paper Scissors'.

