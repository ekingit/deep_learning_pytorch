# 1. CIFAR-10 Image Classification

This folder contains three Jupyter notebooks that explore different deep learning models to image classification on the CIFAR-10 dataset. Each notebook leverages a unique architecture, ranging from a custom convolutional neural network to state-of-the-art pretrained models.

 **Data**

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal of this project is to classify these images into their respective categories using various deep learning models.

**Models**

i) **VGG-16-like Architecture (Built from Scratch)**

- Implements a custom VGG-16-inspired convolutional neural network architecture.
- Designed and trained from scratch for the CIFAR-10 dataset.

ii) **ResNet-18 (Pretrained on ImageNet)**

- Fine-tunes a ResNet-18 model pretrained on the ImageNet dataset for CIFAR-10 classification.
- Trains a randomly initialized ResNet-18 model for comparison.

iii) **Vision Transformer (Pretrained on ImageNet)**

- Fine-tunes a Vision Transformer (ViT) model pretrained on the ImageNet dataset for CIFAR-10 classification.
- Utilizes self-attention mechanisms to capture global image features.
- Adapts the pretrained ViT model to handle the smaller size and specific characteristics of CIFAR-10.
- Demonstrates cutting-edge transformer-based architecture performance on image classification tasks.

# 2. Brain Tumor Classification

This project focuses on classifying brain tumor types using the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Two state-of-the-art architectures, Vision Transformers (ViT) and ResNet, are compared to evaluate their performance.

- **Dataset:** Brain Tumor MRI Dataset
- **Models:**
  - **Vision Transformer (ViT):** Achieves 96.72% accuracy on the test set.
  - **ResNet50:** Achieves 95.7% accuracy on the test set.
- **Pretraining and Training:** Both models use ImageNet pretrained weights and are fine-tuned on the dataset.
