#!/usr/bin/python

import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt



# Linear Neural Network
class Net(nn.Module):
    """
    my first neural network practice

    """
    def __init__(self, n_inputs, n_hidden):
        super().__init__() # to invoke the properties of the parent class nn.Module
        self.in_layer = nn.Linear(n_inputs, n_hidden) # neural activity --> hidden units
        self.out_layer = nn.Linear(n_hidden, 1) #hidden units --> output

    def forward(self, r):
        h = torch.relu(self.in_layer(r))
        y = self.out_layer(h)
        return y


# Fully connect network
class FC(nn.Module):
    def __init__(self, h_in, w_in):
        super().__init__()
        self.dims = h_in * w_in
        self.fc = nn.Linear(self.dims, 10)
        self.out = nn.Linear(10,1)

    def forward(self, x):
        x = x.view(-1, self.dims)
        x = torch.relu(self.fc(x))
        x = torch.sigmoid(self.out(x))
        return x


# Convolutional network
class ConvFC(nn.Module):
    def __init__(self, h_in, w_in):
        super().__init__()
        C_in = 1 # input stimuli have 1 input channel
        C_out = 6 # number of output channels
        K = 7
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=K, padding=K//2) # add padding to avoid error
        self.fc = nn.Linear(np.prod(self.dims), 10)
        self.out = nn.Linear(10,1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = torch.relu(self.conv(x))
        x = x.view(-1, np.prod(self.dims))
        x = torch.relu(self.fc(x))
        x = torch.sigmoid(self.out(x))
        return x


# Max pooling layers
class ConvPoolFC(nn.Module):
    def __init__(self, h_in, w_in):
        super().__init__()
        C_in = 1
        C_out = 6
        K = 7
        Kpool = 8
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=K, padding=K//2)
        self.pool = nn.MaxPool2d(Kpool)
        self.dims = (C_out, h_in // Kpool, w_in // Kpool)
        self.fc = nn.Linear(np.prod(self.dims), 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(-1, np.prod(self.dims))
        x = torch.relu(self.fc(x))
        x = torch.sigmoid(self.out(x))
        return x