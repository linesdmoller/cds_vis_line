# Assignment 1 - Image Search
**Visual Analytics, Cultural Data Science**

By: Line Stampe-Degn Møller

Contributors: None

## PROJECT DESCRIPTION:
*This project is assignment 1 in the sublementary course, 'Visual Analytics' in 'Cultural Data Science', at Aarhus University.*
- Link to assignment description: (https://github.com/CDS-AU-DK/cds-visual/blob/main/assignments/assignment1.md)

**Task outline:**
- Take a user-defined image from the folder
- Calculate the "distance" between the colour histogram of that image and all of the others.
- Find which 3 image are most "similar" to the target image.
- Save an image which shows the target image, the three most similar, and the calculated distance score.
- Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

## METHODS:
The goal of this project was to quantitatively compare images based in their histograms. I approached this task by using OpenCV's calcHist() function and then calculating the 'distance' between two histograms; e.i. the histogram of one preselected target image in the dataset and the histogram of another image in the dataset (looping over each image individually and comparing to target image). The distance between those histogram profiles is calculated using the ChiSquare algorithm, HISTCMP_CHISQR, along with the compareHist() function, both also from OpenCV. I save the filenames of the target image, comparison image, distance and path (to comparison image) in a dataframe that i later sort by the distance scores (lowest to higest) and cut off all other input than the top 3. 
In order to save the target image along with the top 3 comparison images and their distance scores as a png I first add a watermark text to each of the three images displaying their individual distance scores in relation to the target image. I chose to use numpy's horizontal stack, hstck(), and vertical stack, vstack(), functions to stack the three comparison images and the target image in one png. Before stacking, I resize the three comparison images to have the same height for the horizontal stack. I then resize the target image to have the same width as the accumulated width of the three comparison images for the vertical stack. After stacking the images, I save them as one collected png. I also save a txt file of the dataframe showing the listed three comparison images with the closest distance to the target image along with their distance scores.

## USAGE:


## DISCUSSION OF RESULTS:

