import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N = 1000
X = np.random.random((N, 2)) * 6 - 3
Y = np.cos(2*X[:,0]) + np.cos(3*X[:,1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def train(model, criterion, optimizer, X_train, y_train, epochs=1000, print_every=100):

  train_losses = np.zeros(epochs)

  for i in range(epochs):
    epoch = i + 1
    optimizer.zero_grad()

    output = model(X_train)
    loss = criterion(output, y_train)

    loss.backward()
    optimizer.step()

    train_losses[i] = loss.item()

    if epoch % print_every == 0:
      print(f'Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}')
    
  return train_losses

X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(Y.astype(np.float32)).reshape(-1, 1)

train_losses = train(model, loss_function, optimizer, X_train, y_train)

plt.plot(train_losses)