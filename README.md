# DNN
# Functional Network: A Novel Framework for Interpretability of Deep Neural Networks
This is an implementation of functional network for the deep neural network on Python3.

![](./Functional_Network_Of_DNN/doc/pipeline.jpg)

## 0. Requirements

igraph=0.9.11, giotto-tda=0.5.0, PyTorch=1.10.0 or newer, numpy, scikit-learn, scipy

## 1. Preprocess dataset
MNIST, Fashion-MNIST and CIFAR-10 datasets can be downloaded in `./data` via pytorch.

'train_dataset = datasets.MNIST(root='../data', train=True,
                               transform=transforms.ToTensor())'

'test_dataset = datasets.MNIST(root='../data', train=False,
                              transform=transforms.ToTensor())'

## 2. Train neural networks

## 3. Construct the graph model of the functional network

## 4. Construct the simplicial complex model of the functional network
