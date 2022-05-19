# Assignment 1 - Image Search
**Visual Analytics, Cultural Data Science**

By: Line Stampe-Degn Møller

Contributors: None
- Link to this repository: (https://github.com/linesdmoller/cds_vis_line/tree/main/assignment1)

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
In order to save the target image along with the top 3 comparison images and their distance scores as a png I first add a watermark text to each of the three images displaying their individual distance scores in relation to the target image. I chose to use numpy's horizontal stack, hstck(), and vertical stack, vstack(), functions to stack the three comparison images and the target image in one png. Before stacking, I resize the three comparison images to have the same height for the horizontal stack. I then resize the target image to have the same width as the accumulated width of the three comparison images for the vertical stack. After stacking the images, I save them as one collected png (see [Link](https://github.com/linesdmoller/cds_vis_line/blob/main/assignment1/out/image_1304.jpg_comparison_images.png)). I also save a txt file of the dataframe showing the listed three comparison images with the closest distance to the target image along with their distance scores (see [Link](https://github.com/linesdmoller/cds_vis_line/blob/main/assignment1/out/image_1304.jpg_comparison_dataframe.csv)).

## USAGE:
In order to reproduce this project, one must first add the dataset 'flowers' (images) to the 'in' folder. The data can be found in the shared data folder for 'CDS-VIS', under 'flowers'. Then add the folder, 'flowers', inside the 'in' folder, so that the structure in the 'in' folder is; 'in/flowers/image_1304.jpg'.

If one wishes to change the target image, simply type in another filename from the dataset in the 'my_image' variable.

To run this script in the terminal, navigate to the folder outside the 'src' folder and run:

python3 scr/assignment3.py

## DISCUSSION OF RESULTS:
The output of this project is;
- a png of the stacked target image, three closest comparison images and their distance scores.
- a csv of a saved dataframe, outlining the target image, three closest comparison images, their distance scores and their path.

The results show that the three comparison images closest to the target image, 'image_1304.jpg', are;
1. image_0896.jpg
2. image_1115.jpg
3. image_1002.jpg

The png of the images and the distance scores show that the three comparison images are all of yellow flowers. As there is a little yellow on the target image, this kind of makes sense. The CSV of the dataframe is good as it is. However. if one would want to iterate over all images in the dataset as the target image, it might be beneifcial to restructure the dataframe to have only one row for each target image and then have the three comparison images listed in three seperate collumns. This would also require some rethinking of how the code runs, as the current code actively uses paths etc. saved in the current dataframe. However, since both the csv and the png is named after the title of the target image, it is also possible to just iterate and save evry new target image output as a new csv and png file.
