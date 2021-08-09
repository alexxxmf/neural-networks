import torch
import os
import sys
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as transforms
import cv2
from utils.data_fns import create_data_mnist


np.set_printoptions(linewidth=200)

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

X_train, y_train, X_test, y_test = create_data_mnist(FOLDER)

X_train = torch.from_numpy(X_train).type(torch.FloatTensor).reshape(-1, 784)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor).reshape(-1, 784)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

print(y_train.shape)

model = nn.Sequential(
  nn.Linear(784, 32),
  nn.ReLU(),
  nn.Linear(32, 10)
)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.1)

n_epochs = 300

for i in range(n_epochs):
  optimizer.zero_grad()

  outputs = model(X_train)
  loss = loss_function(outputs, y_train)

  loss.item()

  loss.backward()
  optimizer.step()

  print(f'epoch: {i}/{n_epochs}, Loss: {loss.item():.4f}')