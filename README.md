## Car or Bike Image Classification using TensorFlow
This repository contains the code for my project on image classification of car and bike images using TensorFlow. The dataset used for this project is sourced from Kaggle's Car or Bike dataset. To load data: <a href=https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset target="_blank">Car vs Bike Classification Dataset</a>

<!-- [Car vs Bike Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset){:target="_blank"}. -->

Note: To run the code, you will have to download the data and put them in the sub-directories: **data/train**, **data/test** and **data/validation**. These directories will be created if they do not already exist.

### Introduction
The goal of this project is to develop an image classification model that can accurately classify images of cars and bikes. The dataset consists of a collection of car and bike images with labels indicating whether the image is of a car or a bike. I used TensorFlow, an open-source machine learning framework, to build and train a convolutional neural network (CNN) for image classification.

### Dataset
The Car or Bike dataset contains 2000 images of cars and bikes. Due to computational limitations, I only used 600 images for the training phase. The images are of different sizes and aspect ratios. The dataset has been split into training, validating and testing sets.

### Model Architecture
I used a CNN with three convolutional layers, each followed by a max pooling layer, and a final fully connected layer. The model was trained on the training dataset for 20 epochs with a batch size of 8. The model was optimized using binary cross-entropy loss and the Adam optimizer.

### Results
After training the model, I evaluated its performance on the test set. The model achieved an accuracy of 85%, which indicates that it is able to accurately classify car and bike images.

![Prediction of a batch of images](static/prediction_array.png)

### Conclusion
This project demonstrates how to build and train a simple image classification model using TensorFlow. The Car or Bike dataset is a great dataset to start with for image classification tasks. With more data and a more complex model architecture, we could potentially improve the accuracy of the model even further.