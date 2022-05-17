### Assignment 1 - Image Search
# By Line Stampe-Degn Møller
# Visual Analytics, Cultural Data Science

## TASKS:
# - Take a user-defined image from the folder
# - Calculate the "distance" between the colour histogram of that image and all of the others.
# - Find which 3 image are most "similar" to the target image.
# - Save an image which shows the target image, the three most similar, and the calculated distance score.
# - Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

## USAGE:
# To run this script:
# python3 scr/assignment3.py

# IMPORTS:
import os
import sys
import pandas as pd
sys.path.append(os.path.join("..", "..", "CDS-VIS"))
import cv2
import numpy as np
from utils.imutils import jimshow
from utils.imutils import jimshow_channel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


### 1. Take a user-defined image from the folder

# Define file path
my_image = "image_1304.jpg"
filepath = os.path.join("..", "..", "CDS-VIS", "flowers", my_image)

# Load image
target_image = cv2.imread(filepath)
print(f"Image loaded from filepath: {filepath}")

### 2. Calculate the "distance" between the colour histogram of that image and all of the others

# Empty dataframe for saving data
data = pd.DataFrame(columns=["target_image", "comparison_image", "distance", "path"])

# TARGET IMAGE:
# Histogram of target image
target_image_hist = cv2.calcHist([target_image], [0], None, [256], [0,256])
# Normalize
target_image_norm = cv2.normalize(target_image_hist, target_image_hist, 0,255, cv2.NORM_MINMAX)

# ALL THE OTHER IMAGES:
filepath = os.path.join("..", "..", "CDS-VIS", "flowers")
images = os.listdir(filepath)

all_images = []

print("Currently calculating distance scores - this may take a while...")
# Save filepaths of all images (except the target image) in a list
for image in images:
    if image != my_image:
        filepath = os.path.join("..", "..", "CDS-VIS", "flowers", image)
        comparison_image = cv2.imread(filepath)
        all_filepaths = os.path.join(image)
        all_images.append(all_filepaths)

# Calculate distance scores of all images
for image in all_images:
    filepath = os.path.join("..", "..", "CDS-VIS", "flowers", image)
    comparison_image = cv2.imread(filepath)
    # Histogram for comparison_image
    comparison_image_hist = cv2.calcHist([comparison_image], [0], None, [256], [0,256])
    # Normalize
    comparison_image_norm = cv2.normalize(comparison_image_hist, comparison_image_hist, 0, 255, cv2.NORM_MINMAX)
    # Score
    score = round(cv2.compareHist(target_image_norm, comparison_image_norm, cv2.HISTCMP_CHISQR))
    data = data.append({"target_image" : my_image,
                       "comparison_image" : image,
                       "distance" : score,
                       "path" : filepath}, ignore_index = True)
    
print(f"All disance scores calculated")
    
### 3. Find which 3 image are most "similar" to the target image

# Sort values based on distance score
data = data.sort_values("distance", ignore_index = True)
# Only keep top 3
data = data.iloc[:3]

print(f"The 3 most similar to target image:\n,{data}")

### 4. Save an image which shows the target image, the three most similar, and the calculated distance score.

# Save separate list of distance scores of the 3 most similar images
score_list = []
for score in data["distance"]:
    score_list.append(str(score))
    
# Resize comparison images so the height is the same (necessary for the horizontal stack later)
three_images = []
acc_width = 0
image_no = 0

print("IMAGE RESIZING:")
for image in data["path"]:
    img = cv2.imread(image)
    # Get original height and width
    print(f"Original Dimensions of image {image_no}: {img.shape}")
    # resize image by specifying custom width and height
    height = 200    # set value of height
    width = round(((img.shape[1])/(img.shape[0]))*height)   # width reshaped proportionately to the height.
    acc_width = acc_width + width
    resized = cv2.resize(img, (width, height))
    print(f"Resized Dimensions of image {image_no}: {resized.shape}")
    # Add text (distance score) to image:
    ## syntax:
    ## cv2.putText(img, "text", (x, y), font, fontScale, fontColor, thickness, lineType)
    cv2.putText(resized, f"Score: {score_list[image_no]}", (10, 190), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2)
    # Append to list:
    three_images.append(resized)
    # Incresae image count by 1:
    image_no = image_no + 1

# Horizontally stack the 3 images closest to the target image:
hstacked_images = np.hstack(three_images) # horizontal stack

# Vertically stack the target image on top of the 3 comparison images:
# Resize target image to same width as accumulated width of the 3 comparison images (equal width required for vertical stacking)
width = acc_width  # Accumulated width of 3 comparison images
height = round(((img.shape[0])/(img.shape[1]))*width)   # Height reshaped proportionately to the width.
target_image_resized = cv2.resize(target_image, (width, height))
vstacked_images = np.vstack([target_image_resized, hstacked_images]) # vertical stack

# Save stacked image
cv2.imwrite("out/comparison_images.png", vstacked_images)
print("Image saved")

### 5. Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

# Save a csv of dataFrame (created earlier)
data.to_csv(f"out/{my_image}_comparison.csv")
print("csv saved")
