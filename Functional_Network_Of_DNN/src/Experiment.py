import time
import os
import scipy
import torch
import torchtext

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from Model import FCN, train
from Graph_model import Functional_Network
from Simplicial_complex_model import computing_PD


batch_size = 64

# import Fashion-MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True,
                               transform=transforms.ToTensor())

test_dataset = datasets.FashionMNIST(root='./data', train=False,
                              transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


root = "./results/FashionMNIST_[400,400,200]_LeakyReLU_01/"

learning_rate = 3e-4
num_epochs = 100
net_nums = 20
regularizations = [0,0.5,-1]
reg_dirs = ['Vanilla', 'Dropout', 'BatchNorm']


for i in range(len(regularizations)):
    for j in range(net_nums):
        print('This is {:}th Neural Network!'.format(i*net_nums+j+1))
        model, test_acc, test_loss, train_acc, train_loss, net_cor = train(train_dataloader = train_loader, \
             test_dataloader = test_loader, 
             n_neurons=[28*28,400,400,200,10], \
             learning_rate = learning_rate, num_epochs = num_epochs, 
             regularization = regularizations[i])

        # Save the model and cor matrix
        np.savetxt(root+reg_dirs[i]+"/net_cor/{:}th_FNN.txt".format(j),net_cor)
        torch.save(model, root+reg_dirs[i]+"/model/{:}th_FNN.pt".format(j))

# Load the saved cor matrix
net_cors = []
for d in reg_dirs:
    for i in range(net_nums):
        net_cors.append(np.loadtxt(root+d+"/net_cor/" + str(i) + 'th_FNN.txt'))

# Construct the graph model with the density of 5% of the functional network for GTA
graph_models = []
for cor in net_cors:
    graph = Functional_Network(cor=cor, density=0.05)
    graph_models.append(graph)


# Construct the simplicial complex model of the functional network for TDA
net_dgms = computing_PD(net_cors, homology_dimensions=[0,1])

