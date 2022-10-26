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

class FCN(nn.Module):
    # Define the Fully connected network
    
    def __init__(self, neurons, regularization=0):

        # neurons: number of neurons in each layer of the network
        # regularization: the used regularization method;
        # the network is trained with dropout or BatchNorm.
        # if regularization is greater than 0 and less than 1, 
        # train the model with dropout,
        # the dropout rate is set to regularization;
        # if it is equal to 0, train the model without regularization
        # if it is less than 0, train the model with BatchNorm
        
        super(FCN, self).__init__()
        layers = []
        
        for i in range(len(neurons)-2):
            if regularization == 0:
                layers.append(
                    nn.Sequential(
                        nn.Linear(neurons[i], neurons[i+1]),
                        nn.LeakyReLU()
                    )
                )
            elif regularization > 0 and regularization < 1:
                layers.append(
                    nn.Sequential(
                        nn.Linear(neurons[i], neurons[i+1]),
                        nn.Dropout(regularization),
                        nn.LeakyReLU()
                    )
                )
            elif regularization < 0:
                layers.append(
                    nn.Sequential(
                        nn.Linear(neurons[i], neurons[i+1]),
                        nn.BatchNorm1d(neurons[i+1]),
                        nn.LeakyReLU()
                    )
                )
            else:
                raise Exception("dropout_rate must be less than 1")
            
        self.fc_layers = nn.Sequential(*layers)
        self.output = nn.Linear(neurons[-2], neurons[-1])
        
    def forward(self, x):
        
        # Feedforward calculation
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)

            # Record the activation values of hidden neurons
            if i == 0:
                self.features = x
            else:
                self.features = torch.cat((self.features,x), dim=1)
                
        x = self.output(x)
        return x

def train(train_dataloader, test_dataloader, 
             n_neurons = [28*28,300,300,10], learning_rate = 1e-4, regularization = 0,
             num_epochs = 100):
    
    # Train the FCN
    # train_dataloader, test_dataloader: the training dataset and test dataset
    # n_neurons: the network architecture
    # learning_rate: learning rate
    # regularization: dropout of BatchNorm
    # num_epochs: the number of epochs for network training

    model = FCN(neurons = n_neurons, regularization=regularization)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    
    train_acc = np.zeros(num_epochs)
    test_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    test_loss = np.zeros(num_epochs)
    
    layer_output = []
    all_output_cor = None
        
    for epoch in range(num_epochs):
        
        print('epoch{}'.format(epoch+1))
        print('*'*50)
        since = time.time()
        
        running_loss = 0.0
        running_acc = 0.0

        # training
        model.train()
        for i,data in enumerate(train_dataloader,1):
            optimizer.zero_grad()
            
            img,label=data            
            img = img.view(img.size(0),-1)

            if use_gpu:
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)

            # Forward propagation
            out = model(img)
            loss = criterion(out,label)
            running_loss += loss.item()
            _,pred  = torch.max(out,1)
            num_correct = (pred == label)
            running_acc += num_correct.float().mean()

            # Back propagation
            loss.backward()
            optimizer.step()

        train_loss[epoch] = running_loss / len(train_dataloader)
        train_acc[epoch] = running_acc / len(train_dataloader)    
        print(f'Finish {epoch + 1} epoch, Loss: {running_loss / len(train_dataloader):.6f}, Acc: {running_acc / len(train_dataloader):.6f}')

        # testing
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for data in test_dataloader:
            img, label = data
            img = img.view(img.size(0), -1)
            
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(img)
                label = Variable(label)
                
            with torch.no_grad():
                out = model(img)
                
            loss = criterion(out,label)
            eval_loss += loss.item()
            _, pred = torch.max(out, 1)
            eval_acc += (pred == label).float().mean()
        
        print(f'Test Loss: {eval_loss / len(test_dataloader):.6f}, Acc: {eval_acc / len(test_dataloader):.6f}')
        print(f'Time:{(time.time() - since):.1f} s')
        
        test_acc[epoch] = eval_acc / len(test_dataloader)
        test_loss[epoch] = eval_loss / len(test_dataloader)


    # test in the training dataset
    # to record the activation values of hidden neurons
    # for the training dataset
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for data in train_dataloader:
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        with torch.no_grad():
            out = model(img)

        out = model(img)
        loss = criterion(out,label)
        running_loss += loss.item()
        _,pred  = torch.max(out,1)
        num_correct = (pred == label)
        running_acc += num_correct.float().mean()
        
        # record the activation values
        layer_output = layer_output + model.features.data.cpu().numpy().tolist()

    # Compute the cor matrix of hidden neurons
    all_output_value = np.array(layer_output)
    all_output_cor = np.corrcoef(all_output_value, rowvar=False)
    all_output_cor = all_output_cor.astype('float16')
        

    model.features = None
    
    return model, test_acc, test_loss, train_acc, train_loss, all_output_cor
