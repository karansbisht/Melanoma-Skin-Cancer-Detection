# Project Name
 Multiclass classification model using a custom convolutional neural network in TensorFlow. 

# Problem statement:

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

## Table of Contents
*Table of contents
Problem statement:
Create a dataset
Visualize the data
Model Creation
Compile the model
Train the model
Visualizing training results
Model 2 Creation
Compiling the model
Training the model
Visualizing the results
Class Imbalance Detection
Rectify the class imbalance
See the distribution of augmented data after adding new images to the original training data.
Train the model on the data created using Augmentor)
create a training mindset 
create model 
compile model
train your model


## General Information

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
Conclusion 1 from the analysis
- Model is overfitting. From the above Training vs Validation accuracy graph we can see that as the epoch increases the difference between Training accuracy and validation accuracy increases.

Conclusion 2 from the analysis
1. After using data augumentation and dropout layer overfitting issue is reduce.

2. Model Performance is still not increased. Will check the distribution of classes in the training set to check is there have class imbalance.

- Conclusion 3 from the analysis
1. As per the final model (model3) Training accuracy and validation accuracy increases.
2. Model overfitting issue is solved.
3. Class rebalance helps in augmentation and achieving the best Training and validation accuracy.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
#import libraries

import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
pip install argumentor

## Acknowledgements
Give credit here.
- This project was inspired by upgrad
- References if any cancer detect 
- This project was based on [this tutorial](https://www.example.com).


## Contact
Created by [@karansbisht] - feel free to contact me!


<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->