# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam
import utils
import os
#dataTypes = ['digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat']
dataTypes = ['digits-normal.mat']
trainSet = 1
testSet = 2

class CNet(Module):   
    def __init__(self):
        super(CNet, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


for dataType in dataTypes:
    path = os.path.join('..', 'data', dataType)
    data = utils.loadmat(path)
    train_x, train_y = np.transpose(data['x'][:,:,data['set']==trainSet]), data['y'][data['set']==trainSet]
    test_x, test_y = np.transpose(data['x'][:,:,data['set']==testSet]), data['y'][data['set']==testSet]
    #print(train_x.shape, train_y.shape)
    #print(test_x.shape, test_y.shape)

    # converting training and testing images into torch format
    train_x = train_x[:, np.newaxis, :, :]
    test_x = test_x[:, np.newaxis, :, :]
    train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y.astype(int))
    test_x, test_y = torch.from_numpy(test_x).float(), torch.from_numpy(test_y.astype(int))
    # defining the model
    model = CNet().float()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=0.07)
    # defining the loss function
    criterion = CrossEntropyLoss()
    print(model)
    # defining the number of epochs
    n_epochs = 50
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    def train(epoch):
        model.train()
        tr_loss = 0
        # getting the training set
        x_train, y_train = (train_x), (train_y)
        # getting the validation set
        x_val, y_val = (test_x), (test_y)

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        
        # prediction for training and validation set
        output_train = model(x_train)
        output_val = model(x_val)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        loss_val = criterion(output_val, y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        if epoch%2 == 0:
            # printing the validation loss
            print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

    # training the model
    for epoch in range(n_epochs):
        train(epoch)
    # prediction for training set
    with torch.no_grad():
        output = model(train_x)
        
    softmax = torch.exp(output)
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on training set
    print('Training accuracy', accuracy_score(train_y, predictions))
    # prediction for validation set
    with torch.no_grad():
        output = model(test_x)

    softmax = torch.exp(output)
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on validation set
    print('Testing Accuracy', accuracy_score(test_y, predictions))
