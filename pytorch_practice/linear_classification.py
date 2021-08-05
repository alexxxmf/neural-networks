import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TEST_SIZE_PCT = 0.33
N_EPOCHS = 3000
PRINT_EVERY = 30


data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=TEST_SIZE_PCT)

print(f'X shape: {X_train.shape}, y shape: {y_train.shape}')

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
Y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))


model = nn.Sequential(
    nn.Linear(X_train.shape[1], 1),
    nn.Sigmoid()
)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

train_loses = np.zeros(N_EPOCHS)
test_loses = np.zeros(N_EPOCHS)

for i in range(N_EPOCHS):
  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
  # regarding why we need to explicitly set gradients to cero for the optimizer
  optimizer.zero_grad()

  outputs = model(X_train)

  loss = loss_function(outputs, Y_train)

  loss.backward()
  optimizer.step()

  outputs_test = model(X_test)
  loss_test = loss_function(outputs_test, Y_test) 

  train_loses[i] = loss.item()
  test_loses[i] = loss_test.item()

  if (i + 1) % PRINT_EVERY ==0:
    print(f'Epoch {i+1}/{N_EPOCHS}, Train loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

# torch.no_grad() impacts the autograd engine and deactivate it.
# It will reduce memory usage and speed up computations but it s not possible to do backprop\
# this has to do with the fact that by default for every tensor operation autograd acumulates gradients
# for when the derivation time comes (backprop | gradient)
with torch.no_grad():
  p_train = model(X_train)
  p_train = np.round(p_train.numpy())
  train_acc = np.mean(Y_train.numpy() == p_train)

  p_test = model(X_test)
  p_test = np.round(p_test.numpy())
  test_acc = np.mean(Y_test.numpy() == p_test)

print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')