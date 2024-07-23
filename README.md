# Picture Classificatoin on the CIFAR10 Dataset

## Overview
This repository demonstrates an implementation of a Convolutional Neural Network (CNN) for classifying images in the CIFAR-10 dataset. The model architecture is inspired by VGG16 and implemented using PyTorch.


## Aim
The primary objective of this learning project is to develop and train a CNN model to achieve high accuracy in classifying images from the CIFAR-10 dataset into 10 distinct categories. This project serves as a hands-on experience to deepen understanding of deep learning concepts and CNNs.

## Data
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. Each image is a 3x32x32 tensor representing the Red, Green, and Blue (RGB) channels, along with a label ranging from 0 to 9 indicating the class. 

## Data Augmentation
To improve model performance, the training data was augmented using various techniques such as random cropping, horizontal flipping, and normalization. This helps in increasing the diversity of the training data and reducing overfitting.


## Model
The CNN model is inspired by the VGG16 architecture, known for its depth and use of small convolution filters. The model is implemented in PyTorch and consists of multiple convolutional layers followed by max-pooling layers, culminating in fully connected layers. Here is a summary of the model architecture:

## Details

 - Epochs: 50
 - Activation Functions: ReLU (Rectified Linear Unit) in the hidden layers and Log-Softmax for classification
 - Loss Function: Cross-Entropy Loss
 - Optimizer: Stochastic Gradient Descent optimizer (with learning rate=0.01, momentum=0.9)

## Results
The model achieved an accuracy of 85% on the CIFAR-10 test set, demonstrating its effectiveness in image classification tasks.

