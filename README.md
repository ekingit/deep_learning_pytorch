# CIFAR-10 Image Classification

This repository contains three Jupyter notebooks demonstrating different approaches to image classification on the CIFAR-10 dataset. Each approach leverages a unique architecture, ranging from a custom convolutional neural network to state-of-the-art pretrained models.

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal of this project is to classify these images into their respective categories using various deep learning models.

### Notebooks

#### 1. **VGG-16-like Architecture (Built from Scratch)**

- Implements a custom VGG-16-inspired convolutional neural network architecture.
- Designed and trained from scratch for the CIFAR-10 dataset.

#### 2. **ResNet-18 (Pretrained on ImageNet)**

- Fine-tunes a ResNet-18 model pretrained on the ImageNet dataset for CIFAR-10 classification.
- Trains a randomly initialized ResNet-18 model for comparison.

#### 3. **Vision Transformer (Pretrained on ImageNet)**

- Fine-tunes a Vision Transformer (ViT) model pretrained on the ImageNet dataset for CIFAR-10 classification.
- Utilizes self-attention mechanisms to capture global image features.
- Adapts the pretrained ViT model to handle the smaller size and specific characteristics of CIFAR-10.
- Demonstrates cutting-edge transformer-based architecture performance on image classification tasks.

