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
from sklearn.metrics import confusion_matrix
import itertools
from datetime import datetime

train_dataset = torchvision.datasets.FashionMNIST(
    root='./datasets',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./datasets',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

print('Train Dataset shape', train_dataset.data.shape)
print('Test Dataset shape', test_dataset.data.shape)

K = len(set(train_dataset.targets.numpy()))
print(f'Number of classes: {K}')

class CNN(nn.Module):
  def __init__(self, K):
    super(CNN, self).__init__()
    self.conv_layers = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
      nn.ReLU()
    )
    self.dense_layers = nn.Sequential(
      nn.Dropout(0.2),
      # Here is a way to calculate the output of the last conv layer so we can plug it here
      # https://stackoverflow.com/questions/34739151/calculate-dimension-of-feature-maps-in-convolutional-neural-network
      # output dimension for a conv layer is equal to ((n + 2p -f)/s) + 1. If it's decimal we lower it
      # output from last conv layer is a set of 128 feature maps, having each a 2 x 2 dimension
      nn.Linear(2 * 2 * 128, 512),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(512, K),
    )

  def forward(self, X):
    output = self.conv_layers(X)
    flattened_output = output.view(output.size(0), -1)
    return self.dense_layers(flattened_output)

class AlexNet(nn.Module):
  def __init__(self, K):
    super(LeNet5, self).__init__()
    self.conv_layers = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
    )

    self.dense_layers = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, K),
    )
  
  def forward(self, X):
    output = self.conv_layers(X)
    flattened_output = torch.flatten(output, 1)
    return self.dense_layers(flattened_output)

class LeNet5(nn.Module):
  def __init__(self, K):
    super(LeNet5, self).__init__()
    self.conv_layers = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
      # 24 x 24 x 6
      nn.Tanh(),
      nn.AvgPool2d(kernel_size=2),
      # 12 x 12 x 6
      nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
      # 8 x 8 x 16
      nn.Tanh(),
      nn.AvgPool2d(kernel_size=2),
      # 4 x 4 x 16
      nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1),
      # 2 x 2 x 120
      nn.Tanh()
    )

    self.dense_layers = nn.Sequential(
      nn.Linear(2 * 2 * 120, 84),
      nn.Tanh(),
      nn.Linear(84, K)
    )

  def forward(self, X):
    output = self.conv_layers(X)
    flattened_output = torch.flatten(output, 1)
    return self.dense_layers(flattened_output)


model = CNN(K)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

def train(model, criterion, optimizer, train_loader, test_loader, epochs):
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)

  for i in range(epochs):
    t0 = datetime.now()
    train_loss = []

    for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, targets)

      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
    
    train_loss = np.mean(train_loss)

    test_loss = []
    for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      test_loss.append(loss.item())

    test_loss = np.mean(test_loss)

    train_losses[i] = train_loss
    test_losses[i] = test_loss

    dt = datetime.now() - t0
    print(f'Epoch {i+1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Test Loss: {test_loss:.4f}, Duration: {dt}')

  return train_losses, test_losses

train_losses, test_losses = train(
  model, loss_function, optimizer, train_loader, test_loader, epochs=15)

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

n_correct = 0.
n_total = 0.

for inputs, targets in train_loader:
  inputs, targets = inputs.to(device), targets.to(device)

  outputs = model(inputs)

  _, predictions = torch.max(outputs, 1)

  n_correct += (predictions == targets).sum().item()
  n_total += targets.shape[0]

train_acc = n_correct / n_total

for inputs, targets in test_loader:
  inputs, targets = inputs.to(device), targets.to(device)

  outputs = model(inputs)

  _, predictions = torch.max(outputs, 1)

  n_correct += (predictions == targets).sum().item()
  n_total += targets.shape[0]

test_acc = n_correct / n_total

print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}') 

def plot_confusion_matrix(cm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
  
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print("Confusion matrix, without normalization")

  print(cm)
  
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j], fmt,
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()
p_test = np.array([])
for inputs, targets in test_loader:
  inputs, targets = inputs.to(device), targets.to(device)
  
  outputs = model(inputs)

  _, predictions = torch.max(outputs, 1)

  p_test = np.concatenate((p_test, predictions.cpu().numpy()))

cm = confusion_matrix(y_test, p_test)

plot_confusion_matrix(cm, list(range(10)))