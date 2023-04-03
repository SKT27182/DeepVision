# DeepVision

Implementing Computer Vision Architectures from Scratch using TensorFlow or PyTorch

## Introduction

This repository contains implementations of various Computer Vision Architectures from scratch using TensorFlow or PyTorch. The implementations are done in a modular fashion, so that the individual components of the architectures can be used in other projects.

## Architectures

## Classification Models

###  [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf): This is the first Convolutional Neural Network (CNN) architecture proposed by Yann LeCun in 1998. It was used to classify handwritten digits. The architecture consists of three convolutional layers, two subsampling layers, and three fully connected layers. For more details, refer to the [LeNet](Notes/classification/LeNet.md) notes.

---

###  [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf): This is the first CNN architecture to win the ImageNet competition in 2012. It was proposed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. The architecture consists of five convolutional layers, three pooling layers, and two fully connected layers. For more details, refer to the [AlexNet](Notes/classification/AlexNet.md) notes.

---

### [VGG](https://arxiv.org/pdf/1409.1556v6.pdf): This is the first CNN architecture to use very deep convolutional layers. It was proposed by Karen Simonyan and Andrew Zisserman. The architecture consists of 16 convolutional layers, 5 pooling layers, and 3 fully connected layers. For more details, refer to the [VGG](Notes/classification/VGG.md) notes.

---

### [ResNet](https://arxiv.org/pdf/1512.03385.pdf): This is the first CNN architecture to use residual connections. It was proposed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. The architecture consists of 18 convolutional layers, 5 pooling layers, and 3 fully connected layers. For more details, refer to the [ResNet](Notes/classification/ResNet.md) notes.

---
---

## Generative Models

### [AutoEncoder](https://www.science.org/doi/pdf/10.1126/science.1127647): This is a neural network architecture that is used to learn efficient data encodings in an unsupervised manner. It was proposed by Geoffrey Hinton and his students at the University of Toronto. The architecture consists of two parts: an encoder and a decoder. The encoder learns to compress the input data into a lower dimensional representation, and the decoder learns to reconstruct the input data from the lower dimensional representation. For more details, refer to the [AutoEncoder](Notes/generative/Autoencoder.md) notes.

---