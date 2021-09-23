#!/usr/bin/python

import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


####################################################################
####################################################################
# Pytorch Tutorial

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor,
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("shape of X [N,C,H,W]: ", X.shape)
    print("shape of y: ", y.shape, y.dtype)
    break


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)






##############################################################################################################################
##############################################################################################################################
# basic pratical session of torch

# rand1 = torch.rand(5,5)
# rand2 = torch.rand(5,5)
# print(rand1)

# # get value
# a = rand1[2,2].item()
# print(a)

# # matrix multiplication
# rand_mm = rand1.mm(rand2)
# print(rand_mm)

# #Matrix product of two tensors.if 1-d, same with np.dot; if 2-d, same with torch.mm
# b = rand1.matmul(rand2[:, 0])


# # in-place operation
# rand11 = rand1.add_(1)
# rand12 = rand1.div_(2)
# rand13 = rand1.zero_()

# # dimenions
# ## torch.randn is from normal distribution
# ab = torch.randn(10,10)
# print(ab.unsqueeze(-1).size())
# print(ab.unsqueeze(0).size())
# print(ab.unsqueeze(1).size())
# print(ab.unsqueeze(-1).squeeze(1).size())


# # different dimensions
# a = torch.randn(2)
# print(a)
# a = a.unsqueeze(-1)
# print(a)

# # expand across dimensions
# print(a.expand(2,3))

# print(torch.cuda.is_available()) # true



# pratical session ends
################################################################################################################################
################################################################################################################################

# # Linear Neural Network
# class Net(nn.Module):
#     """
#     my first neural network practice

#     """
#     def __init__(self, n_inputs, n_hidden):
#         super().__init__() # to invoke the properties of the parent class nn.Module
#         self.in_layer = nn.Linear(n_inputs, n_hidden) # neural activity --> hidden units
#         self.out_layer = nn.Linear(n_hidden, 1) #hidden units --> output

#     def forward(self, r):
#         h = torch.relu(self.in_layer(r))
#         y = self.out_layer(h)
#         return y


# # Fully connect network
# class FC(nn.Module):
#     def __init__(self, h_in, w_in):
#         super().__init__()
#         self.dims = h_in * w_in
#         self.fc = nn.Linear(self.dims, 10)
#         self.out = nn.Linear(10,1)

#     def forward(self, x):
#         x = x.view(-1, self.dims)
#         x = torch.relu(self.fc(x))
#         x = torch.sigmoid(self.out(x))
#         return x


# # Convolutional network
# class ConvFC(nn.Module):
#     def __init__(self, h_in, w_in):
#         super().__init__()
#         C_in = 1 # input stimuli have 1 input channel
#         C_out = 6 # number of output channels
#         K = 7
#         self.conv = nn.Conv2d(C_in, C_out, kernel_size=K, padding=K//2) # add padding to avoid error
#         self.fc = nn.Linear(np.prod(self.dims), 10)
#         self.out = nn.Linear(10,1)

#     def forward(self, x):
#         x = x.unsqueeze(1) 
#         x = torch.relu(self.conv(x))
#         x = x.view(-1, np.prod(self.dims))
#         x = torch.relu(self.fc(x))
#         x = torch.sigmoid(self.out(x))
#         return x


# # Max pooling layers
# class ConvPoolFC(nn.Module):
#     def __init__(self, h_in, w_in):
#         super().__init__()
#         C_in = 1
#         C_out = 6
#         K = 7
#         Kpool = 8
#         self.conv = nn.Conv2d(C_in, C_out, kernel_size=K, padding=K//2)
#         self.pool = nn.MaxPool2d(Kpool)
#         self.dims = (C_out, h_in // Kpool, w_in // Kpool)
#         self.fc = nn.Linear(np.prod(self.dims), 10)
#         self.out = nn.Linear(10, 1)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = torch.relu(self.conv(x))
#         x = self.pool(x)
#         x = x.view(-1, np.prod(self.dims))
#         x = torch.relu(self.fc(x))
#         x = torch.sigmoid(self.out(x))
#         return x