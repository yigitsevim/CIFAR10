# CIFAR10

## Introduction:

This code is a convolutional neural network implementation for the **CIFAR10 dataset**. The CIFAR10 dataset is a collection of **60,000 32x32 color images in 10 classes, with 6,000 images per class**. There were initially 50,000 training images and 10,000 test images. However, to prevent overfitting and keeping a better track of the training process I randomly selected 10,000 images from training set to create a validation set. Then, I augmented 30,000 images from the remaining training set and added the augmented images to trainin set. So in total, I have **70,000** training images, **10,000** validation images and **10,000** test images. The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The implementation uses **PyTorch** deep learning framework, which is a widely used and well-documented framework for deep learning. The code consists of the following main parts:

**Data Preprocessing:** The CIFAR10 dataset is loaded and preprocessed using torchvision.transforms module, which applies normalization and data augmentation transforms to the images.

**Model Definition:** The model architecture is defined using the nn.Module class of PyTorch. **The model consists of 2 convolutional layers, followed by 2 fully connected layers**.

**Training:** The model is trained using the Adam optimizer and Cross-Entropy loss function. The training is done for 200 epochs on a GPU.

**Evaluation:** The accuracy of the trained model is evaluated on the test set.

The code achieves a **test accuracy of around 86%**, which is a good result considering the complexity of the CIFAR10 dataset.
